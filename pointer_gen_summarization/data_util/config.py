import os

ROOT_DIR = "/u/nlp/data/cnn_dm_2/"  # where all information is contained
LOG_ROOT = os.path.join(ROOT_DIR, "dataset/log/")  # where all log data for training jobs will be contained, e.g. model checkpoints, evaluation results, and decoding outputs.
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "dataset/finished_files/chunked/train_*")  # path to chunked training data
EVAL_DATA_PATH = os.path.join(ROOT_DIR, "dataset/finished_files/val.bin")
DECODE_DATA_PATH = os.path.join(ROOT_DIR, "dataset/finished_files/test.bin")
VOCAB_PATH = os.path.join(ROOT_DIR, "dataset/finished_files/vocab")  # default vocab path

CUSTOM_VOCAB_PATH = "/Users/alexshan/Desktop/pointer_gen_summarization/pointer_gen_summarization/data_util/glove.pt"

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 4
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=100000
lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0
eps = 1e-12
max_iterations = 500000
use_gpu=True
lr_coverage=0.15
