# Summarization with Pointer Generator Networks
Python3 PyTorch implementation and [stanfordnlp/stanza]([https://github.com/stanfordnlp/stanza]) of Pointer-Generator Networks for Abstractive Summarization by See et. al.

Integration with Stanza, Stanford NLP's open-source library for many human languages. Load your own custom embeddings and use our Character Language Model embeddings for this task! 

This code is a Stanza-integrated version of the paper [Get To The Point: Summarization with Pointer-Generator Networks]([url](https://arxiv.org/abs/1704.04368)). I have also updated the implementation to work with PyTorch 2.1 and removed many of the deprecated implementation details related to Tensorflow 1.13. 

## Additions ##
In this repo, you can load custom word embeddings to train the abstractive summarization model. You can also bring your own character language model embeddings from Stanza to this repo!
Additionally, you can compute the ROUGE scores for the results using the `rouge-score` library instead of the `pyrouge` library implemented in the original paper, which has since changed.
Finally, we've added the capacity to compute semantic similarity scores (BERTScore and S-BERT embedding cosine similarity) across the generated summaries and reference text.

## Running the code
**Important**: when running experiments, make sure your configurations for hyperparameters and paths are set properly in `data_util/config.py`


Using the `training_ptr_gen/` directory, the `train.py` module allows you to train a model on the dataset from scratch. 
The `eval.py` module or `eval_all.py` modules allow you to run the model checkpoint(s) against the validation set. This can be run concurrently to the training script!
Finally, once your model is done training, you can run the `decode.py` script to decode the results of the model using beam search. After decoding the generated summaries, you can score them using either ROUGE or semantic similarity scoring methods. To compute ROUGE scores, use the `rouge_scoring/score_summaries.py` script. For semantic similarity scores, see `semantic_evals/run_semantic_eval.py`.

Stanza: https://github.com/stanfordnlp/stanza

`Get To The Point: Summarization with Pointer-Generator Networks` by See et. al.: https://arxiv.org/abs/1704.04368
