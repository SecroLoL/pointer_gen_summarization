from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO)

import tensorflow as tf
import torch
from model import Model
from typing import List, Tuple, Mapping
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad

from data_util import config
from data_util.batcher import Batcher, Batch
from data_util.data import Vocab, load_custom_vocab
from data_util.utils import calc_running_avg_loss
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch
from training_ptr_gen.eval import Evaluate

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self, custom_vocab_path: str = "", charlm_forward_file: str = "", charlm_backward_file: str = ""):
        """
        Constructor for a training job object

        Loads in custom word embeddings if provided
        Loads in charlm embeddings if provided

        Loads the training dataset
        Creates training dirs to log data
        """
        # Load custom Vocabulary, if needed
        self.use_custom_vocab = os.path.exists(custom_vocab_path)
        self.custom_word_embedding = None  # by default, use standard embeddings
        self.custom_vocab_path = custom_vocab_path
        if self.use_custom_vocab:
            custom_vocab, custom_emb = load_custom_vocab(vocab_path=custom_vocab_path)
            print(f"Using custom word embeddings taken from path {custom_vocab_path}.")
            self.custom_word_embedding = custom_emb
            self.vocab = custom_vocab
        else:  # use default vocab list
            self.vocab = Vocab(config.VOCAB_PATH, config.vocab_size)
            print(f"Using base Vocab from path {config.VOCAB_PATH} and size {config.vocab_size}.")
        # Load charlm embeddings if provided
        self.charlm_forward_file = charlm_forward_file
        self.charlm_backward_file = charlm_backward_file
        self.use_charlm = os.path.exists(charlm_forward_file) and os.path.exists(charlm_backward_file)
        if self.use_charlm:
            print(f"Using charlm files {charlm_forward_file} and {charlm_backward_file}.")
        if self.use_custom_vocab and self.custom_word_embedding is None or self.custom_word_embedding is not None and not self.custom_word_embedding:
            raise ValueError(f"The value of self.use_custom_vocab ({self.use_custom_vocab}) and "
                             f"self.custom_word_embedding ({self.custom_word_embedding}) are incompatible.")
        # Load data from Dataset Batcher
        self.batcher = Batcher(data_path=config.TRAIN_DATA_PATH, 
                               vocab=self.vocab, 
                               mode='train',
                               batch_size=config.batch_size, 
                               single_pass=False)
        print(f"Loading batches using training data from {config.TRAIN_DATA_PATH}")
        time.sleep(15)
        # Load data dirs 
        train_dir = os.path.join(config.LOG_ROOT, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        print(f"Using train dir {train_dir}")

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        print(f"Using model_dir {self.model_dir}")
        self.summary_writer = tf.summary.create_file_writer(train_dir)

    def save_model(self, running_avg_loss: float, iter: int):
        """
        Saves model state dict, optimizer, and current loss.

        Args:
            running_avg_loss (float): The running average loss for the current training job
            iter (int): The number of completed training iterations
        
        """
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        print(f"Saving model to {model_save_path}")
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path: str = None):
        """
        Creates model from `model_file_path`, or creates it from scratch to a new file if not provided.

        Creates optimizer, sets up save file path, and loads in current loss and iteration number
        """
        print(f"Executing setup train with {model_file_path if model_file_path is not None else 'a new model.'}")

        self.model = Model(
            model_file_path,
            custom_word_embedding=self.custom_word_embedding,
            charlm_forward_file=self.charlm_forward_file,
            charlm_backward_file=self.charlm_backward_file,
        )
    

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
        print(f"Starting on iter #: {start_iter} with loss {start_loss}.")

        return start_iter, start_loss

    def train_one_batch(self, batch: Batch):
        """
        Executes training across a single batch of examples.
        """
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, truncated_articles = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)
        self.optimizer.zero_grad()
        
        # enc batch is the tensor of shape (B, seq len) with the token IDs of the article
        # Using the article words themselves (remember, it takes [[str]]) so we need to split each article
        # into sentences because the build_char_reps function takes lists of sentences
        # The length of truncated_articles should be the batch size

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens, truncated_articles)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,coverage, di)
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

        # Compute losses
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)
        loss.backward()
        # Clip gradient norm
        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def trainIters(self, n_iters: int, model_file_path: str = None, SAVE_EVERY: int = 5000):
        """
        Trains model for `n_iters`, loading model from `model_file_path` if provided.

        Note that `n_iters` is an absolute number. So that means that if the model has already trained for 1000 iterations,
        setting `n_iters` to 2000 will train it for another 1000 iterations. 

        If `model_file_path` not provided, this function will create one.

        After every `SAVE_EVERY` iterations, the model params will be saved.

        """
        # Validate the model path
        if model_file_path is not None and not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file path provided ({model_file_path}) could not be found. Aborting.")
        if model_file_path is None:
            print("Beginning model training from scratch.")
        else:
            print(f"Beginning training with model from {model_file_path}")
        
        # Initialize the model and restore parameters if a model file path was specified.
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        # Execute train job
        print(f"Finished training setup. Beginning train: iteration {iter} / {n_iters}")
        while iter < n_iters:
            batch = self.batcher.next_batch()  # load the next training batch
            loss = self.train_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1
            # Log training progress
            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % SAVE_EVERY == 0:   # save and evaluate model every 5000 iters
                self.save_model(running_avg_loss, iter)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
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
    
    train_processor = Train(custom_vocab_path=args.custom_vocab_path,
                            charlm_forward_file=args.charlm_forward_file,
                            charlm_backward_file=args.charlm_backward_file)
    train_processor.trainIters(config.max_iterations, args.model_file_path)
