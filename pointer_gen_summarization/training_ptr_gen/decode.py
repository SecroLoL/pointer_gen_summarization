#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys
import argparse
import os
import time

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab, load_custom_vocab
from data_util import data, config
from training_ptr_gen.model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from training_ptr_gen.train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path: str, custom_vocab_path: str, charlm_forward_file: str, charlm_backward_file: str):
        print(f"creating beam searcher for model {model_file_path}")
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.LOG_ROOT, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        print(f"Decode dir: {self._decode_dir}")
        print(f"ROUGE REF DIR: {self._rouge_ref_dir}")

        self.use_custom_vocab = os.path.exists(custom_vocab_path)
        self.custom_word_embedding = None  # by default, use standard embeddings
        if self.use_custom_vocab:
            print(f"Creating custom Vocab with path {custom_vocab_path} and size {config.vocab_size}.")
        else:
            print(f"Using base Vocab from path {config.VOCAB_PATH} and size {config.vocab_size}.")

        if self.use_custom_vocab:
            custom_vocab, custom_emb = load_custom_vocab(vocab_path=custom_vocab_path)
            print(f"Using custom word embeddings taken from path {custom_vocab_path}.")
            self.custom_word_embedding = custom_emb
            self.vocab = custom_vocab
        else:  # use default vocab list
            self.vocab = Vocab(config.VOCAB_PATH, config.vocab_size)
        
        self.charlm_forward_file = charlm_forward_file
        self.charlm_backward_file = charlm_backward_file
        self.use_charlm = os.path.exists(charlm_forward_file) and os.path.exists(charlm_backward_file)
        if self.use_charlm:
            print(f"Using charlm files {charlm_forward_file} and {charlm_backward_file}.")

        self.batcher = Batcher(config.DECODE_DATA_PATH, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        print(f"Data from {config.DECODE_DATA_PATH}")
        time.sleep(15)

        if self.use_custom_vocab and self.custom_word_embedding is None or self.custom_word_embedding is not None and not self.custom_word_embedding:
            raise ValueError(f"The value of self.use_custom_vocab ({self.use_custom_vocab}) and "
                            f"self.custom_word_embedding ({self.custom_word_embedding}) are incompatible.")

        self.model = Model(model_file_path, 
                           is_eval=True,
                           custom_word_embedding=self.custom_word_embedding,
                           charlm_forward_file=charlm_forward_file,
                           charlm_backward_file=charlm_backward_file)


    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self, finished_decoding: bool = False):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        if not finished_decoding:
            while batch is not None:
                # Run beam search to get best Hypothesis
                best_summary = self.beam_search(batch)

                # Extract the output ids from the hypothesis and convert back to words
                output_ids = [int(t) for t in best_summary.tokens[1:]]
                decoded_words = data.outputids2words(output_ids, self.vocab,
                                                    (batch.art_oovs[0] if config.pointer_gen else None))

                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words

                original_abstract_sents = batch.original_abstracts_sents[0]

                write_for_rouge(original_abstract_sents, decoded_words, counter,
                                self._rouge_ref_dir, self._rouge_dec_dir)
                counter += 1
                if counter % 1000 == 0:
                    print('%d example in %d sec'%(counter, time.time() - start))
                    start = time.time()

                batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0, truncated_articles = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens, truncated_articles)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    
    parser.add_argument("--custom_vocab_path",
                        dest="custom_vocab_path",
                        required=False,
                        default="",
                        help="Optional custom vocab path to a PT file containing a custom vocabulary.")
    parser.add_argument("--charlm_forward_file",
                        dest="charlm_forward_file",
                        required=False,
                        default=None,
                        help="Optional custom charlm forward model file.")
    
    parser.add_argument("--charlm_backward_file",
                        dest="charlm_backward_file",
                        required=False,
                        default=None,
                        help="Optional custom charlm backward model file.")

    args = parser.parse_args()

    model_filename = args.model_file_path
    custom_vocab_path = args.custom_vocab_path
    charlm_forward, charlm_backward = args.charlm_forward_file, args.charlm_backward_file

    print(f"running beam search decoding on model from path {model_filename}")
    beam_Search_processor = BeamSearch(model_filename, 
                                       custom_vocab_path,
                                       charlm_forward_file=charlm_forward,
                                       charlm_backward_file=charlm_backward
                                       )
    beam_Search_processor.decode()


