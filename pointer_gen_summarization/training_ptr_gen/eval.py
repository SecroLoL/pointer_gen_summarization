from __future__ import unicode_literals, print_function, division

import argparse
import os
import logging
import time
import sys

import tensorflow as tf
import torch

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch
from training_ptr_gen.model import Model
from data_util.data import load_custom_vocab

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path, custom_vocab_path: str, charlm_forward_file: str, charlm_backward_file: str):
        print(f"Creating evaluator for model in path {model_file_path}")
        self.use_custom_vocab = os.path.exists(custom_vocab_path)
        self.custom_word_embedding = None  # by default, use standard embeddings
        if self.use_custom_vocab:
            custom_vocab, custom_emb = load_custom_vocab(vocab_path=custom_vocab_path)
            print(f"Using custom word embeddings taken from path {custom_vocab_path}.")
            self.custom_word_embedding = custom_emb
            self.vocab = custom_vocab
        else:  # use default vocab list
            self.vocab = Vocab(config.VOCAB_PATH, config.vocab_size)
            print(f"Using base Vocab from path {config.VOCAB_PATH} and size {config.vocab_size}.")

        self.charlm_forward_file = charlm_forward_file
        self.charlm_backward_file = charlm_backward_file
        self.use_charlm = os.path.exists(charlm_forward_file) and os.path.exists(charlm_backward_file)
        if self.use_charlm:
            print(f"Using charlm files {charlm_forward_file} and {charlm_backward_file}.")
        
        print(f"Eval data path : {config.EVAL_DATA_PATH}")
        self.batcher = Batcher(config.EVAL_DATA_PATH, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.LOG_ROOT, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)

        print(f"Eval dir {eval_dir}")
        self.summary_writer = tf.summary.create_file_writer(eval_dir)
        if self.use_custom_vocab and self.custom_word_embedding is None or self.custom_word_embedding is not None and not self.custom_word_embedding:
            raise ValueError(f"The value of self.use_custom_vocab ({self.use_custom_vocab}) and "
                            f"self.custom_word_embedding ({self.custom_word_embedding}) are incompatible.")

        self.model = Model(model_file_path, 
                           is_eval=True,
                           custom_word_embedding=self.custom_word_embedding,
                           charlm_forward_file=charlm_forward_file,
                           charlm_backward_file=charlm_backward_file)

        

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, truncated_articles = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens, truncated_articles)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.item()

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        count = 0
        while batch is not None:
            count += 1
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                iter, print_interval, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()
        
        print(f"Finished running eval on {count} batches.")


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

    print(f"Running eval on model {model_filename}...")
    eval_processor = Evaluate(
        model_filename, 
        custom_vocab_path, 
        charlm_forward_file=charlm_forward, 
        charlm_backward_file=charlm_backward
    )
    eval_processor.run_eval()
