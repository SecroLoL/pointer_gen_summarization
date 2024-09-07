import numpy as np
import torch
from data_util import config
from data_util.batcher import Batch

def get_input_from_batch(batch: Batch, use_cuda: bool):
  """
  Given a Batch object, extract information needed for training on batch
  """
  batch_size = len(batch.enc_lens)

  enc_batch = torch.from_numpy(batch.enc_batch).long()
  enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).float()
  enc_lens = batch.enc_lens
  extra_zeros = None
  enc_batch_extend_vocab = None

  if config.pointer_gen:
    enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).long()
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
      extra_zeros = torch.zeros((batch_size, batch.max_art_oovs))

  c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

  coverage = None
  if config.is_coverage:
    coverage = torch.zeros(enc_batch.size())

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()

    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
    if extra_zeros is not None:
      extra_zeros = extra_zeros.cuda()
    c_t_1 = c_t_1.cuda()

    if coverage is not None:
      coverage = coverage.cuda()

  # Attempt to load in the truncated text
  original_articles = batch.original_articles
  truncated_articles = []
  for art in original_articles:
    art_words = art.decode().split()
    art_words = art_words[: config.max_enc_steps]
    truncated_articles.append(art_words)
    
  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, truncated_articles

def get_output_from_batch(batch: Batch, use_cuda: bool):
  """
  Get the output tensors from a batch of data.

  The outputs for a batch are the target summaries 
  Args:
    batch (Batch): The input batch containing the data.
    use_cuda (bool): Flag indicating whether to use CUDA for tensor operations.
  Returns:
    tuple: A tuple containing the following tensors:
      - dec_batch (torch.Tensor): The decoded batch tensor.
      - dec_padding_mask (torch.Tensor): The padding mask tensor for the decoded batch.
      - max_dec_len (int): The maximum length of the decoded sequences.
      - dec_lens_var (torch.Tensor): The tensor containing the lengths of the decoded sequences.
      - target_batch (torch.Tensor): The target batch tensor.
  """
  dec_batch = torch.from_numpy(batch.dec_batch).long()
  dec_padding_mask = torch.from_numpy(batch.dec_padding_mask).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = torch.from_numpy(dec_lens).float()

  target_batch = torch.from_numpy(batch.target_batch).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()
  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch