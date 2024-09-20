# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py

import queue
import logging 
import time
import random
import numpy as np
import tensorflow as tf
from threading import Thread
from typing import List, Tuple, Mapping
import data_util.config as config
import data_util.data as data

random.seed(1234)
logger = logging.getLogger(__name__)


class Example(object):
  """
  Attributes:
    enc_len (int): The length of the processed article after truncation but before padding.
    enc_input (List[int]): The list of word ids representing the processed article. Out-of-vocabulary (OOV) words are represented by the id for the UNK token.
    dec_input (List[int]): The input sequence for the decoder.
    target (List[int]): The target sequence for the decoder.
    dec_len (int): The length of the decoder input sequence.
    enc_input_extend_vocab (List[int]): The version of the enc_input where in-article OOVs are represented by their temporary OOV id.
    article_oovs (List[str]): The in-article OOV words themselves.
    original_article (str): The original article text.
    original_abstract (str): The original abstract text.
    original_abstract_sents (List[str]): The original abstract text, split into sentences. One sentence per list item.
  Methods:
    get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id): Generate input and target sequences for the decoder.
    pad_decoder_inp_targ(max_len, pad_id): Pad the decoder input and target sequences with a specified padding ID up to a maximum length.
    pad_encoder_input(max_len, pad_id):
      Pad the encoder input sequences with a specified pad_id up to a maximum length.
  """
  def __init__(self, article: str, abstract_sentences: List[str], vocab):
    """
    Constructor for an Example object.

    Args:
        article (str): The article text.
        abstract_sentences (List[str]): The abstract text, split into sentences.
        vocab (Vocab): A Vocab object.
    """
    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)

    # Process the article
    article_words = article.decode().split()
    if len(article_words) > config.max_enc_steps:
      article_words = article_words[:config.max_enc_steps]
    self.enc_len = len(article_words) # store the length after truncation but before padding
    self.enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token

    # Process the abstract
    abstract = ' '.encode().join(abstract_sentences).decode() # string
    abstract_words = abstract.split() # list of strings
    abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if config.pointer_gen:
      # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
      self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

      # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
      abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    self.original_article = article  # str
    self.original_abstract = abstract  # str
    self.original_abstract_sents = abstract_sentences  # List[str]. One sentence per list item


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    """
    Generate input and target sequences for decoder.
    Args:
      sequence (list): The input sequence.
      max_len (int): The maximum length of the sequences.
      start_id (int): The start token ID.
      stop_id (int): The stop token ID.
    Returns:
      tuple: A tuple containing the input sequence and the target sequence.
    """
    
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    """
    Pad the decoder input and target sequences with a specified padding ID up to a maximum length.
    Parameters:
      max_len (int): The maximum length of the sequences.
      pad_id (int): The padding ID to be used for padding.
    Returns:
      None
    """
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


  def pad_encoder_input(self, max_len, pad_id):
    """
    Pads the encoder input sequences with a specified pad_id up to a maximum length.
    Args:
      max_len (int): The maximum length of the encoder input sequences.
      pad_id (int): The pad_id used for padding the sequences.
    Returns:
      None
    """
    
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    if config.pointer_gen:
      while len(self.enc_input_extend_vocab) < max_len:
        self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
  def __init__(self, example_list, vocab, batch_size):
    """
    Constructor for a Batch object from a list of examples.

    Compiles the list of Example objects into a single Batch.
    """
    self.batch_size = batch_size
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list) # initialize the input to the encoder
    self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings


  def init_encoder_seq(self, example_list: List[Example]):
    """
    Creates an input sequence to the encoder 

    Args:
        example_list (List[Example]): A list of Example objects
    """
    # Find the longest input seq in the batch for encoder
    max_enc_seq_len = max([ex.enc_len for ex in example_list])

    # Pad all examples to make them uniform length
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    # enc_batch is a tensor of shape (B, seq len) that contains the token ID for each token (word)
    self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)  
    self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)  # length of each example
    self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)  # representing what is padded or not

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]  # for each example, fill in the batch with the token IDs
      # ex.enc_input is the list of token IDs for each token in the input sequence
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if config.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list: List[Example]):
    """
    Creates a input and target sequence for the decoder.

    Args:
        example_list (List[Example]): A list of Example objects of the target batch
    """
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    """
    Store the original article and abstract strings in the Batch object

    Args:
      example_list (List[Example]): A list of Example objects
    """
    self.original_articles = [ex.original_article for ex in example_list] # List[str]
    self.original_abstracts = [ex.original_abstract for ex in example_list] # List[str]
    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # List[List[str]]

class Batcher(object):
  """
  A class that handles batching of examples for training or decoding.

  Args:
    data_path (str): The path to the data file.
    vocab (Vocab): The vocabulary object.
    mode (str): The mode of operation ('train', 'eval', or 'decode').
    batch_size (int): The batch size.
    single_pass (bool): Whether to read the data file only once.
  
  Attributes:
    BATCH_QUEUE_MAX (int): The maximum number of batches the batch_queue can hold.
    _data_path (str): The path to the data file.
    _vocab (Vocab): The vocabulary object.
    _single_pass (bool): Whether to read the data file only once.
    mode (str): The mode of operation ('train', 'eval', or 'decode').
    batch_size (int): The batch size.
    _batch_queue (Queue): A queue of Batches waiting to be used.
    _example_queue (Queue): A queue of Examples waiting to be batched.
    _num_example_q_threads (int): The number of threads to fill the example queue.
    _num_batch_q_threads (int): The number of threads to fill the batch queue.
    _bucketing_cache_size (int): The number of batches-worth of examples to load into cache before bucketing.
    _finished_reading (bool): Indicates whether we have finished reading the dataset.
    _example_q_threads (list): A list of threads that fill the example queue.
    _batch_q_threads (list): A list of threads that fill the batch queue.
    _watch_thread (Thread): A thread that watches the other threads and restarts them if they're dead.
  
  Methods:
    next_batch(): Retrieves the next batch from the batch queue.
    fill_example_queue(): Fills the example queue with Examples.
    fill_batch_queue(): Fills the batch queue with Batches.
    watch_threads(): Watches the other threads and restarts them if they're dead.
    text_generator(): Generates text from the example generator.
  """
  BATCH_QUEUE_MAX = 25 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, mode, batch_size, single_pass):
    """
    Initializes a Batcher object.
    Args:
      data_path (str): The path to the data.
      vocab: The vocabulary object.
      mode (str): The mode of operation.
      batch_size (int): The batch size.
      single_pass (bool): Whether to run in single pass mode or not.
    """
    self._data_path = data_path
    self._vocab = vocab
    self._single_pass = single_pass
    self.mode = mode
    self.batch_size = batch_size
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 1 #16 # num threads to fill example queue
      self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
      self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def next_batch(self):
    """
    Retrieves the next batch from the batch queue.
    """
    try:
      # If the batch queue is empty, print a warning
      if self._batch_queue.qsize() == 0:
        logger.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
        if self._single_pass and self._finished_reading:
          logger.info("Finished reading dataset in single_pass mode.")
          return None

      batch = self._batch_queue.get() # get the next Batch
      return batch
    except Exception as e:
      print(f"Exception in next_batch: {str(e)}")
      return None 

  def fill_example_queue(self):
    """
    Fills the example queue with examples from the data source.

    This method reads examples from a file and processes them into Example objects. It uses a text generator to iterate through the examples. Each example consists of an article and an abstract, both represented as strings. The method continues filling the example queue until there are no more examples left in the data source.
    If the single_pass mode is on and there are no more examples, the method sets the _finished_reading flag to True and stops. If the single_pass mode is off and there are no more examples, the method raises an exception.
    The abstract is processed into a list of sentences using the <s> and </s> tags. Each sentence is stripped of leading and trailing whitespace.
    After processing an example, it is placed in the example queue for further processing.
    
    Parameters:
    - self: The Batcher object.
    Returns:
    - None
    """
    try:
      input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))
    except Exception as e:
      print(f'in fill_example_queue: {str(e)}')

    while True:
      try:
        (article, abstract) = input_gen.__next__() # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        logger.info("The example generator for this example queue filling thread has exhausted data.")
        print("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          logger.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          print("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
      example = Example(article, abstract_sentences, self._vocab) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.

  def fill_batch_queue(self):
    """
    Fills the batch queue with batches of Examples.
    If the mode is 'decode', it creates a batch with a single example repeated multiple times.
    Otherwise, it retrieves batches of Examples from the example queue, sorts them based on the length of the encoder sequence,
    groups them into batches, optionally shuffles the batches, and places them in the batch queue.
    
    Returns:
      None
    """
    
    while True:
      if self.mode == 'decode':
        # beam search decode mode single example repeated in the batch
        ex = self._example_queue.get()
        b = [ex for _ in range(self.batch_size)]
        self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
      else:
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in range(self.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self.batch_size):
          batches.append(inputs[i:i + self.batch_size])
        if not self._single_pass:
          random.shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

  def watch_threads(self):
    """
    Monitors the status of the example and batch queues in a continuous loop.
    Prints the size of the bucket queue and input queue.
    Restarts any dead threads in the example and batch queues.
    Returns:
      None
    """
    while True:
      logger.info(
        'Bucket queue size: %i, Input queue size: %i',
        self._batch_queue.qsize(), self._example_queue.qsize())

      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          logger.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          logger.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()

  def text_generator(self, example_generator):
    """
    Generates text data from an example generator.
    Args:
      example_generator: An iterator that generates tf.Example objects.
    Yields:
      A tuple containing the article text and abstract text from each tf.Example.
    Raises:
      ValueError: If the article or abstract cannot be retrieved from the example.
    Warnings:
      If an example has an empty article text, it will be skipped.
    """
    
    while True:
      e = example_generator.__next__() # e is a tf.Example
      if e is None: 
        print("e is None")
        raise ValueError("e is None")
      try:
        article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
      except ValueError:
        logger.error('Failed to get article or abstract from example')
        continue
      if len(article_text) == 0: # See https://github.com/abisee/pointer-generator/issues/1
        logger.warning('Found an example with empty article text. Skipping it.')
        continue
      else:
        yield (article_text, abstract_text)
