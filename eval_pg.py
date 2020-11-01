import glob

import numpy as np
import os
import time
from tqdm import tqdm

import tensorflow as tf
from collections import namedtuple

import data
import util
from data import Vocab
from batcher import Batcher, create_batch
from model import SummarizationModel
from decode import BeamSearchDecoder, decode_example
import convert_data
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill
from absl import app, flags, logging
import random

random.seed(222)
FLAGS = flags.FLAGS

# Where to find data
flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm} or a custom dataset name')
flags.DEFINE_string('data_root', os.path.expanduser('~') + '/data/tf_data/with_coref', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
flags.DEFINE_string('vocab_path', 'logs/vocab', 'Path expression to text vocabulary file.')
flags.DEFINE_string('pretrained_path', 'logs/cnn_dm_sent_singles', 'Directory of pretrained model for PG trained on singles or pairs of sentences.')
flags.DEFINE_boolean('use_pretrained', True, 'If True, use pretrained model in the path FLAGS.pretrained_path.')

# Where to save output
flags.DEFINE_string('log_root', 'logs', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', 'pg_lambdamart', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Don't change these settings
flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
flags.DEFINE_boolean('single_pass', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('actual_log_root', '', 'Dont use this setting, only for internal use. Root directory for all logging.')
flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val, test}')

# Hyperparameters
flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
flags.DEFINE_integer('batch_size', 16, 'minibatch size')
flags.DEFINE_integer('max_enc_steps', 100, 'max timesteps of encoder (max source text tokens)')
flags.DEFINE_integer('max_dec_steps', 30, 'max timesteps of decoder (max summary tokens)')
flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
flags.DEFINE_integer('min_dec_steps', 10, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
flags.DEFINE_float('lr', 0.15, 'learning rate')
flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

# PG-MMR settings
flags.DEFINE_boolean('pg_mmr', False, 'If true, use the PG-MMR model.')
flags.DEFINE_string('importance_fn', 'tfidf', 'Which model to use for calculating importance. Must be one of {svr, tfidf, oracle}.')
flags.DEFINE_float('lambda_val', 0.6, 'Lambda factor to reduce similarity amount to subtract from importance. Set to 0.5 to make importance and similarity have equal weight.')
flags.DEFINE_integer('mute_k', 7, 'Pick top k sentences to select and mute all others. Set to -1 to turn off.')
flags.DEFINE_boolean('retain_mmr_values', False, 'Only used if using mute mode. If true, then the mmr being\
                            multiplied by alpha will not be a 0/1 mask, but instead keeps their values.')
flags.DEFINE_string('similarity_fn', 'rouge_l', 'Which similarity function to use when calculating\
                            sentence similarity or coverage. Must be one of {rouge_l, ngram_similarity}')
flags.DEFINE_boolean('plot_distributions', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')

flags.DEFINE_string('singles_and_pairs', 'singles',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_string('ssi_exp_name', 'lambdamart_singles',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')

_exp_name = 'lambdamart'
_dataset_articles = 'cnn_dm'
dataset_split = 'test'
num_instances = -1
random_seed = 123
# singles_and_pairs = 'both'
start_over = True

num_test_examples = 14490

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
temp_dir = 'data/temp/scores'
lambdamart_in_dir = 'data/temp/to_lambdamart'
lambdamart_out_dir = 'data/temp/lambdamart_results'
ssi_out_dir = 'data/temp/ssi'
log_dir = 'logs'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json')]


def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.singles_and_pairs == 'both':
        FLAGS.exp_name = FLAGS.exp_name + '_both'
        exp_name = _exp_name + '_both'
        dataset_articles = _dataset_articles
    else:
        FLAGS.exp_name = FLAGS.exp_name + '_singles'
        exp_name = _exp_name + '_singles'
        dataset_articles = _dataset_articles + '_singles'
    my_log_dir = os.path.join(log_dir, FLAGS.ssi_exp_name)



    print('Running statistics on %s' % FLAGS.exp_name)

    if FLAGS.dataset_name != "":
        FLAGS.data_path = os.path.join(FLAGS.data_root, FLAGS.dataset_name, FLAGS.dataset_split + '*')
    if not os.path.exists(os.path.join(FLAGS.data_root, FLAGS.dataset_name)) or len(os.listdir(os.path.join(FLAGS.data_root, FLAGS.dataset_name))) == 0:
        print(('No TF example data found at %s so creating it from raw data.' % os.path.join(FLAGS.data_root, FLAGS.dataset_name)))
        convert_data.process_dataset(FLAGS.dataset_name)

    logging.set_verbosity(logging.INFO) # choose what level of logging you want
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.exp_name = FLAGS.exp_name if FLAGS.exp_name != '' else FLAGS.dataset_name
    FLAGS.actual_log_root = FLAGS.log_root
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
                   'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps',
                   'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'lambdamart_input']
    hps_dict = {}
    for key,val in FLAGS.__flags.items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val.value # add it to the dict
    hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

    tf.set_random_seed(113) # a seed value for randomness

    decode_model_hps = hps._replace(
        max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

    with open(os.path.join(my_log_dir, 'ssi.pkl')) as f:
        ssi_list = pickle.load(f)

    total = len(source_files) * 1000 if 'cnn' or 'newsroom' in dataset_articles else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False,
                                               should_check_valid=False)
    # batcher = Batcher(None, vocab, hps, single_pass=FLAGS.single_pass)
    model = SummarizationModel(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(model, None, vocab)
    decoder.decode_iteratively(example_generator, total, names_to_types, ssi_list, hps)


    a=0

if __name__ == '__main__':
    app.run(main)

























