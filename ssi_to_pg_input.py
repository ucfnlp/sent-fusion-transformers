import glob

import numpy as np
import os

import util
from decode import BeamSearchDecoder
from absl import app, flags
import random

random.seed(222)
FLAGS = flags.FLAGS

# Where to find data
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm} or a custom dataset name')
flags.DEFINE_string('data_root', 'data', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
flags.DEFINE_string('vocab_path', 'logs/vocab', 'Path expression to text vocabulary file.')
flags.DEFINE_string('pretrained_path', '', 'Directory of pretrained model for PG trained on singles or pairs of sentences.')
flags.DEFINE_boolean('use_pretrained', True, 'If True, use pretrained model in the path FLAGS.pretrained_path.')

# Where to save output
flags.DEFINE_string('log_root', 'logs', 'Root directory for all logging.')
flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Don't change these settings
flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
flags.DEFINE_boolean('single_pass', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
flags.DEFINE_string('actual_log_root', '', 'Dont use this setting, only for internal use. Root directory for all logging.')
flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val, test}')

# Hyperparameters
flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')

if 'batch_size' not in flags.FLAGS:
    flags.DEFINE_integer('batch_size', 16, 'minibatch size')

if 'max_enc_steps' not in flags.FLAGS:
    flags.DEFINE_integer('max_enc_steps', 100, 'max timesteps of encoder (max source text tokens)')

if 'max_dec_steps' not in flags.FLAGS:
    flags.DEFINE_integer('max_dec_steps', 60, 'max timesteps of decoder (max summary tokens)')

if 'beam_size' not in flags.FLAGS:
    flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
if 'min_dec_steps' not in flags.FLAGS:
    flags.DEFINE_integer('min_dec_steps', 10, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_integer("num_instances", -1, "How many examples to run on. -1 means to run on all of them.")
flags.DEFINE_string('original_dataset_name', '',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')

flags.DEFINE_boolean('unilm_decoding', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'coref' not in flags.FLAGS:
    flags.DEFINE_boolean('coref', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'coref_dataset' not in flags.FLAGS:
    flags.DEFINE_boolean('coref_dataset', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_string("resolver", "spacy", "Which method to use for turning token tag probabilities into binary tags. Can be one of {threshold, summ_limit, inst_limit}.")
if 'coref_head' not in flags.FLAGS:
    flags.DEFINE_integer('coref_head', 4, 'Beam size for beam search')
if 'coref_layer' not in flags.FLAGS:
    flags.DEFINE_integer('coref_layer', 4, 'Beam size for beam search')
if 'link' not in flags.FLAGS:
    flags.DEFINE_boolean('link', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'poc_dataset' not in flags.FLAGS:
    flags.DEFINE_boolean('poc_dataset', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('concatbaseline', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'first_chain_only' not in flags.FLAGS:
    flags.DEFINE_boolean('first_chain_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'first_mention_only' not in flags.FLAGS:
    flags.DEFINE_boolean('first_mention_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')


random_seed = 123
# singles_and_pairs = 'both'
start_over = True

num_test_examples = 11490

temp_dir = 'data/temp/scores'
lambdamart_in_dir = 'data/temp/to_lambdamart'
lambdamart_out_dir = 'data/temp/lambdamart_results'
ssi_out_dir = 'data/temp/ssi'
log_dir = 'logs'

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    extractor = 'bert'
    pretrained_dataset = FLAGS.dataset_name
    if FLAGS.coref_dataset or FLAGS.poc_dataset:
        FLAGS.pretrained_path = os.path.join(FLAGS.log_root, 'cnn_dm_crd_both')
    elif FLAGS.singles_and_pairs == 'both':
        FLAGS.pretrained_path = os.path.join(FLAGS.log_root, pretrained_dataset + '_both')
    if FLAGS.singles_and_pairs == 'both':
        FLAGS.exp_name = FLAGS.dataset_name + '_' + FLAGS.exp_name + '_' + extractor + '_both'
        dataset_articles = FLAGS.dataset_name
    if FLAGS.coref_dataset:
        FLAGS.exp_name += '_crd'
    if FLAGS.poc_dataset:
        FLAGS.exp_name += "_pocd"
    if FLAGS.unilm_decoding:
        FLAGS.exp_name += 'unilm'
    if FLAGS.coref:
        FLAGS.exp_name += '_coref'
        FLAGS.exp_name += '_l%d_h%d' % (FLAGS.coref_layer, FLAGS.coref_head)
    if FLAGS.link:
        FLAGS.exp_name += '_link'
    if FLAGS.first_chain_only:
        FLAGS.exp_name += '_fc'
    if FLAGS.first_mention_only:
        FLAGS.exp_name += '_fm'


    ssi_list = None


    print('Exp_name: %s' % FLAGS.exp_name)

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.exp_name = FLAGS.exp_name if FLAGS.exp_name != '' else FLAGS.dataset_name
    FLAGS.actual_log_root = FLAGS.log_root
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    print(util.bcolors.OKGREEN + "Experiment path: " + FLAGS.log_root + util.bcolors.ENDC)

    vocab_datasets = [os.path.basename(file_path).split('vocab_')[1] for file_path in glob.glob(FLAGS.vocab_path + '_*')]
    original_dataset_name = [file_name for file_name in vocab_datasets if file_name in FLAGS.dataset_name]
    if len(original_dataset_name) > 1:
        raise Exception('Too many choices for vocab file')
    if len(original_dataset_name) < 1:
        raise Exception('No vocab file for dataset created. Run make_vocab.py --dataset_name=<my original dataset name>')
    original_dataset_name = original_dataset_name[0]
    FLAGS.original_dataset_name = original_dataset_name

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    np.random.seed(random_seed)
    source_dir = os.path.join('data', 'input_decoding')
    if FLAGS.poc_dataset:
        source_dir += '_pocd'
    else:
        source_dir += '_crd'
    source_data_path = os.path.join(source_dir, FLAGS.dataset_split + '.tsv')

    model = None
    decoder = BeamSearchDecoder(model, None)
    decoder.decode_iteratively(source_data_path, num_test_examples, ssi_list)


if __name__ == '__main__':
    app.run(main)

























