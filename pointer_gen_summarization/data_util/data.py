import glob
import random
import struct
import csv
import torch
import torch.nn as nn
from typing import List, Tuple, Mapping
from tensorflow.core.example import example_pb2
from stanza.models.common.foundation_cache import load_pretrain
from data_util import config

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
  
  """
  Vocab class represents a vocabulary used for word-to-id and id-to-word mappings.

  Attributes:
    _word_to_id (dict): A dictionary that maps words to their corresponding ids.
    _id_to_word (dict): A dictionary that maps ids to their corresponding words.
    _count (int): The total number of words in the vocabulary.
  Methods:
    __init__(self, vocab_file, max_size, use_pt: bool = False, pt_vocab = None):
      Initializes the Vocab object with either a vocabulary file or a PT vocab object.
    word2id(self, word):
      Converts a word to its corresponding id.
    id2word(self, word_id):
      Converts an id to its corresponding word.
    size(self):
      Returns the size of the vocabulary.
    write_metadata(self, fpath):
      Writes the word embedding metadata file to the specified file path.

  """

  def __init__(self, vocab_file: str, max_size: int, use_pt: bool = False, pt_vocab = None):
    """
    Vocab constructor.

    A Vocab can be created with either a vocabulary file, where each line is a word separated from a number by a space.

    Or, a PT vocab object can be provided as well.
    """
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1
    if use_pt:  # use PT vocab object to create vocab, not a vocab file
      print(f"Attempting to create Vocab with a custom PT file")
      for w in pt_vocab._unit2id:
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print(f"max_size of vocab was specified as {max_size}; we now have {self._count} words. Stopping reading.")
          break
    else:
      print(f"Attempting to create vocab for file {vocab_file}. Not using PT file.")
      # Read the vocab file and add words up to max_size
      with open(vocab_file, 'r') as vocab_f:
        for line in vocab_f:
          pieces = line.split()
          if len(pieces) != 2:
            print(f"Warning: incorrectly formatted line in vocabulary file: {line}\n")
            continue
          w = pieces[0]
          if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
          if w in self._word_to_id:
            raise Exception('Duplicated word in vocabulary file: %s' % w)
          self._word_to_id[w] = self._count
          self._id_to_word[self._count] = w
          self._count += 1
          if max_size != 0 and self._count >= max_size:
            print(f"max_size of vocab was specified as {max_size}; we now have {self._count} words. Stopping reading.")
            break
    
    print(f"Finished constructing vocabulary of {self._count} total words. Last word added: {self._id_to_word[self._count-1]}")

  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def write_metadata(self, fpath):
    print(f"Writing word embedding metadata file to {fpath}...") 
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in range(self.size()):
        writer.writerow({"word": self._id_to_word[i]})

  def __len__(self):
    return self.size()


def example_generator(data_path: str, single_pass: bool) -> example_pb2.Example:
  """
    This function is a generator that yields examples from data files.
  Parameters:
  - data_path (str): The path to the data files.
  - single_pass (bool): If True, the function will read the data files only once. If False, the function will shuffle the data files and read them repeatedly.
  Yields:
  - example_pb2.Example: An example object parsed from the data files.
  Note:
  - The function assumes that the data files are in a specific format and can be read using the `example_pb2.Example` protocol buffer.
  """
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)
    if single_pass:
      print("example_generator completed reading all datafiles. No more data.")
      # self._finished_reading = True
      # break
      return None 


def article2ids(article_words: List[str], vocab: Vocab) -> Tuple[List[int], List[str]]:
  """
  Converts a list of article words into a list of corresponding word IDs using the provided vocabulary.
  Parameters:
  - article_words (list): A list of words representing an article.
  - vocab (Vocabulary): An instance of the Vocabulary class.
  Returns:
  - ids (list): A list of word IDs corresponding to the article words.
  - oovs (list): A list of out-of-vocabulary (OOV) words found in the article.
  """
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs


def abstract2ids(abstract_words: List[str], vocab: Vocab, article_oovs: List[str]) -> List[int]:
  """
  Convert a list of abstract words to a list of corresponding word ids using the given vocabulary.
  Args:
    abstract_words (list): A list of abstract words.
    vocab (Vocabulary): The vocabulary object containing word-to-id mappings.
    article_oovs (list): A list of out-of-vocabulary words found in the article.
  Returns:
    list: A list of word ids corresponding to the abstract words.
  """
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in article_oovs: # If w is an in-article OOV
        vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list: List[int], vocab: Vocab, article_oovs: List[str]) -> List[str]:
  """
  Convert a list of word IDs to a list of words using the provided vocabulary and article OOVs.
  Args:
    id_list (list): A list of word IDs.
    vocab (Vocabulary): The vocabulary object containing word-to-ID mappings.
    article_oovs (list): A list of out-of-vocabulary words in the article.
  Returns:
    list: A list of words corresponding to the given word IDs.
  Raises:
    ValueError: If the model produces a word ID that isn't in the vocabulary and article_oovs is None.
    ValueError: If the model produces a word ID that corresponds to an article OOV that doesn't exist in the given article_oovs.
  """
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e: # i doesn't correspond to an article oov
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words


def abstract2sents(abstract: str) -> List[str]:
  """
  Convert an abstract into a list of sentences.
  Args:
    abstract (str): The abstract to be converted.
  Returns:
    list: A list of sentences extracted from the abstract.
  """
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START.encode(), cur)
      end_p = abstract.index(SENTENCE_END.encode(), start_p + 1)
      cur = end_p + len(SENTENCE_END.encode())
      sents.append(abstract[start_p+len(SENTENCE_START.encode()):end_p])
    except ValueError as e: # no more sentences
      return sents


def show_art_oovs(article: str, vocab: Vocab) -> str:
  """
  Show out-of-vocabulary (OOV) words in the given article.
  Args:
    article (str): The article text.
    vocab (Vocabulary): The vocabulary object.
  Returns:
    str: The article text with OOV words marked.
  """
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = []
  for w in words:
    if vocab.word2id(w) == unk_token:
      words.append(f"__{w}__")
    else:
      words.append(w)
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(abstract: str, vocab: Vocab, article_oovs: List[str]) -> str:
  """
  Show out-of-vocabulary (OOV) words in the given abstract.

  Args:
    abstract (str): The abstract text.
    vocab (Vocab): The vocabulary object.
    article_oovs (list): A list of out-of-vocabulary words in the article.

  Returns:
    str: The abstract text with OOV words marked.
  """
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is OOV
      if article_oovs is None: # baseline mode
        new_words.append(f"__{w}__")
      else: # pointer-generator mode
        if w in article_oovs:
          new_words.append(f"__{w}__")
        else:
          new_words.append(f"!!__{w}__!!")
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str


def load_custom_vocab(vocab_path: str) -> Tuple[Vocab, nn.Embedding]:
  """
  Creates a Vocab object from a vocab path that is a .pt file, so it contains more than just the lines and words.

  Need to generate the vocab file mapping

  Also need to ensure the embedding matrix lines up with that. 
  The PT Embedding has a .vocab attribute that gives the (i, word) pairs for each vocab
  The PT embedding also has a emb_matrix of shape (vocab size, emb dim) where each word has its embedding.
  
  So what we can do is to have the vocab file transferred from the vocab object. This gives us a file list of the words
  we load that into the Vocab constructor. It will create the vocab object that is identical to the original one, but 
  the indices will be shifted over by 4, since [0, 1, 2, 3] are all occupied. 

  For the PT embedding, we can also load that into the pretrain, but then it has to be modified slightly because we need
  custom embeddings for the first 4 words. We can simply extend the matrix to be (vocab size + 4, emb dim) in shape.
  """

  """
  vocab_path (str): The path to the .pt vocab file

  Returns a tuple of (Vocab object, Embedding object) corresponding to the custom vocab object
  """

  print(f"Loading a custom Vocab object with the PT path {vocab_path}")

  EXTRA_TOKENS = [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]

  pt = load_pretrain(vocab_path)
  pt_vocab = pt.vocab
  vocab = Vocab(vocab_file="", 
                max_size=config.vocab_size,  # will read this many tokens from the vocab.
                use_pt=True, 
                pt_vocab=pt_vocab)
  
  emb_matrix = pt.emb  # (vocab size, emb dim)
  # we may need to truncate the embedding matrix if certain words are getting pushed out of the vocab size limits. This will make room for the extra tokens, and ensure that all embeddings are synced to their updated indices in the vocab.
  emb_matrix = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True).weight.data[: config.vocab_size-len(EXTRA_TOKENS), :]  # get just the emb matrix weights for the tokens that we will actually be using (truncating to make room for EXTRA tokens if needed)
  vocab_size, emb_dim = emb_matrix.shape   
  # We need to extend the emb matrix at the front to account for the new tokens in EXTRA_TOKENS
  extra_tokens_tensor = torch.zeros(len(EXTRA_TOKENS), emb_dim)  # embeddings for EXTRA_TOKENS
  new_embedding_tensor = torch.cat((extra_tokens_tensor, emb_matrix), dim=0)  # now size (vocab_size + len(EXTRA_TOKENS), emb dim)
  new_embedding = nn.Embedding(vocab_size + len(EXTRA_TOKENS), emb_dim)
  new_embedding.weight.data = new_embedding_tensor  # transfer weights over
  return vocab, new_embedding


if __name__ == "__main__":
  vocab, embedding = load_custom_vocab(vocab_path="/Users/alexshan/Desktop/pointer_gen_summarization/pointer_gen_summarization/data_util/glove.pt")
 
