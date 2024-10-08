import os

ROOT_DIR = os.getenv("PGEN_ROOT_DIR", "/u/nlp/data/cnn_dm_2/")  # where all information is contained
LOG_ROOT = os.getenv("PGEN_LOG_ROOT", os.path.join(ROOT_DIR, "dataset/log/"))  # where all log data for training jobs will be contained, e.g. model checkpoints, evaluation results, and decoding outputs.
TRAIN_DATA_PATH = os.getenv("PGEN_TRAIN_DATA_PATH", os.path.join(ROOT_DIR, "dataset/finished_files/chunked/train_*"))  # path to chunked training data
EVAL_DATA_PATH = os.getenv("PGEN_EVAL_DATA_PATH", os.path.join(ROOT_DIR, "dataset/finished_files/chunked/val_*"))  # path to chunked validation data
DECODE_DATA_PATH = os.getenv("PGEN_DECODE_DATA_PATH", os.path.join(ROOT_DIR, "dataset/finished_files/test_*"))  # path to chunked test data
VOCAB_PATH = os.getenv("PGEN_VOCAB_PATH", os.path.join(ROOT_DIR, "dataset/finished_files/vocab"))  # default vocab path

# Train Hyperparameters
hidden_dim = 256
emb_dim = 100
batch_size = 8
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 100000
lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0
eps = 1e-12
max_iterations = 500000
use_gpu = True
lr_coverage = 0.15
