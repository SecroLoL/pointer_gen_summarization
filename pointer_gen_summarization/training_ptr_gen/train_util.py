from torch.autograd import Variable
import numpy as np
import torch
from data_util import config

def get_input_from_batch(batch, use_cuda: bool):
  """
  Given a Batch object, extract information needed for training on batch
  """
  batch_size = len(batch.enc_lens)

  # TODO use this to implement charlm embeddings
  """
  batch.original_articles is a List[str], one string for each article in the batch.

  make sure to check whether these original articles get truncated. Or else, we have to 
  truncate it as well from article words. From batcher.py: `article_words = article.decode().split()`

  So we can use the same thing and we would have the same word tokenization.
  From there, we can truncate the articles to List[str] where each string is the truncated article in the batch.

  Then, we include the truncated articles in the output of this function. From there, we load it into the
  model encoder with the build_char_reps function.
  """

  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens
  extra_zeros = None
  enc_batch_extend_vocab = None

  if config.pointer_gen:
    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

  coverage = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))

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

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch, use_cuda):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()


  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

