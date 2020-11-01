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
import pickle
from absl import app, flags, logging
import random

random.seed(222)
FLAGS = flags.FLAGS

# Where to find data
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm} or a custom dataset name')
flags.DEFINE_string('data_root', os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens', 'Path to root directory for all datasets (already converted to TensorFlow examples).')
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
flags.DEFINE_float('lr', 0.15, 'learning rate')
flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
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

flags.DEFINE_boolean('attn_vis', False, 'If true, then output attention visualization during decoding.')

if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
# flags.DEFINE_string('ssi_exp_name', 'lambdamart_singles',
#                     'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_boolean('upper_bound', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('cnn_dm_pg', False, 'If true, use PG trained on CNN/DM for testing.')
flags.DEFINE_boolean('websplit', False, 'If true, use PG trained on Websplit for testing.')
flags.DEFINE_boolean('use_bert', True, 'If true, use PG trained on Websplit for testing.')
if 'sentemb' not in flags.FLAGS:
    flags.DEFINE_boolean('sentemb', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'artemb' not in flags.FLAGS:
    flags.DEFINE_boolean('artemb', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'plushidden' not in flags.FLAGS:
    flags.DEFINE_boolean('plushidden', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
flags.DEFINE_integer("num_instances", -1, "How many examples to run on. -1 means to run on all of them.")
flags.DEFINE_boolean('skip_with_less_than_3', True,
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_string('original_dataset_name', '',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_string('ssi_data_path', '',
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_boolean('better_beam_search', False,
                    'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_boolean('word_imp_reg', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_float('imp_loss_wt', 0.5, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
flags.DEFINE_boolean('imp_loss_oneminus', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('first_intact', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'tag_tokens' not in flags.FLAGS:
    flags.DEFINE_boolean('tag_tokens', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('by_instance', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('sep', False, 'If true, add a separator token [SEP] between sentences.')
if 'tag_loss_wt' not in flags.FLAGS:
    flags.DEFINE_float("tag_loss_wt", 0.2, "Whether to use TPU or GPU/CPU.")
if 'tag_threshold' not in flags.FLAGS:
    flags.DEFINE_float("tag_threshold", 0.2, "What threshold to use for choosing tags. Only applicable if binarize_method == 'threshold'")
flags.DEFINE_float("summ_limit", 100, "What limit to use for choosing tags. Only applicable if binarize_method == 'summ_limit'")
flags.DEFINE_float("inst_limit", 10, "What limit to use for choosing tags. Only applicable if binarize_method == 'inst_limit'")
flags.DEFINE_string("binarize_method", "summ_limit", "Which method to use for turning token tag probabilities into binary tags. Can be one of {threshold, summ_limit, inst_limit}.")
flags.DEFINE_boolean('only_tags', False, 'If true, add a separator token [SEP] between sentences.')
flags.DEFINE_integer("ninst", 4, "How many instances to grab from BERT to write the summary. -1 means to keep grabbing instances until the 100 token limit is reached.")
flags.DEFINE_boolean('upper_bound_ssi_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('word_eval_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')

flags.DEFINE_boolean('unilm_decoding', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_integer('num_workers', 8, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_integer('worker_idx', -1, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'coref' not in flags.FLAGS:
    flags.DEFINE_boolean('coref', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'coref_dataset' not in flags.FLAGS:
    flags.DEFINE_boolean('coref_dataset', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_string("resolver", "spacy", "Which method to use for turning token tag probabilities into binary tags. Can be one of {threshold, summ_limit, inst_limit}.")
if 'coref_head' not in flags.FLAGS:
    flags.DEFINE_integer('coref_head', 4, 'Beam size for beam search')
if 'coref_layer' not in flags.FLAGS:
    flags.DEFINE_integer('coref_layer', 4, 'Beam size for beam search')
if 'repl' not in flags.FLAGS:
    flags.DEFINE_boolean('repl', False, 'Beam size for beam search')
flags.DEFINE_boolean('pull_from_decode_dir', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('compare_iter_rouge', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_string('compare_exp_name', 'cnn_dm_crd_unilm_bert_both_summ100.0_ninst4', 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('corr_tests', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'allcorefs' not in flags.FLAGS:
    flags.DEFINE_boolean('allcorefs', False, '')
flags.DEFINE_boolean('only_rouge', False, '')
if 'link' not in flags.FLAGS:
    flags.DEFINE_boolean('link', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'link_use_sep' not in flags.FLAGS:
    flags.DEFINE_boolean('link_use_sep', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'poc_dataset' not in flags.FLAGS:
    flags.DEFINE_boolean('poc_dataset', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'poc_gold' not in flags.FLAGS:
    flags.DEFINE_boolean('poc_gold', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('concatbaseline', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'link_invis' not in flags.FLAGS:
    flags.DEFINE_boolean('link_invis', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'first_chain_only' not in flags.FLAGS:
    flags.DEFINE_boolean('first_chain_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'first_mention_only' not in flags.FLAGS:
    flags.DEFINE_boolean('first_mention_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'triples_only' not in flags.FLAGS:
    flags.DEFINE_boolean('triples_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')


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

    if FLAGS.poc_dataset:
        names_to_types = [('raw_article_sents', 'string_list'), ('summary_text', 'string'), ('coref_chains', 'delimited_list_of_list_of_lists')]
    elif FLAGS.triples_only:
        names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'),
                          ('summary_text', 'string'),
                          ('article_lcs_paths_list', 'delimited_list_of_list_of_lists'),
                          ('coref_chains', 'delimited_list_of_list_of_lists')]
    elif FLAGS.coref_dataset:
        # names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'),
        #                   ('article_lcs_paths_list', 'delimited_list_of_list_of_lists'), ('coref_chains', 'delimited_list_of_list_of_lists'), ('coref_representatives', 'string_list')]
        names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'),
                          ('article_lcs_paths_list', 'delimited_list_of_list_of_lists'), ('coref_chains', 'delimited_list_of_list_of_lists')]
    else:
        names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]


    extractor = 'bert' if FLAGS.use_bert else 'lambdamart'
    if FLAGS.cnn_dm_pg:
        pretrained_dataset = 'cnn_dm'
    elif FLAGS.websplit:
        pretrained_dataset = 'websplit'
    else:
        pretrained_dataset = FLAGS.dataset_name
    if FLAGS.dataset_name == 'duc_2004':
        pretrained_dataset = 'cnn_dm'
    if FLAGS.coref_dataset or FLAGS.poc_dataset:
        FLAGS.pretrained_path = os.path.join(FLAGS.log_root, 'cnn_dm_crd_both')
    elif FLAGS.singles_and_pairs == 'both':
        FLAGS.pretrained_path = os.path.join(FLAGS.log_root, pretrained_dataset + '_both')
    else:
        FLAGS.pretrained_path = os.path.join(FLAGS.log_root, pretrained_dataset + '_singles')
    if FLAGS.singles_and_pairs == 'both':
        FLAGS.exp_name = FLAGS.dataset_name + '_' + FLAGS.exp_name + '_' + extractor + '_both'
        dataset_articles = FLAGS.dataset_name
    else:
        FLAGS.exp_name = FLAGS.dataset_name + '_' + FLAGS.exp_name + '_' + extractor + '_singles'
        dataset_articles = FLAGS.dataset_name + '_singles'
    if FLAGS.coref_dataset:
        FLAGS.exp_name += '_crd'
    if FLAGS.poc_dataset:
        FLAGS.exp_name += "_pocd"
        if FLAGS.poc_gold:
            FLAGS.exp_name += '_pocgold'
    if FLAGS.unilm_decoding:
        FLAGS.exp_name += 'unilm'
    elif FLAGS.concatbaseline:
        FLAGS.exp_name += 'concatbaseline'
    else:
        FLAGS.exp_name += 'pg'
    if FLAGS.coref:
        FLAGS.exp_name += '_coref'
        FLAGS.exp_name += '_l%d_h%d' % (FLAGS.coref_layer, FLAGS.coref_head)
    if FLAGS.link_use_sep:
        FLAGS.exp_name += '_use_sep'
    if FLAGS.link:
        FLAGS.exp_name += '_link'
        if FLAGS.link_invis:
            FLAGS.exp_name += '_invis'
    if FLAGS.first_chain_only:
        FLAGS.exp_name += '_fc'
    if FLAGS.first_mention_only:
        FLAGS.exp_name += '_fm'
    if FLAGS.repl:
        FLAGS.exp_name += '_repl'
    if FLAGS.allcorefs:
        FLAGS.exp_name += '_allcorefs'
    if FLAGS.triples_only:
        FLAGS.exp_name += '_triples'
    if FLAGS.word_imp_reg:
        FLAGS.pretrained_path += '_imp' + str(FLAGS.imp_loss_wt)
        FLAGS.exp_name += '_imp' + str(FLAGS.imp_loss_wt)
        if FLAGS.imp_loss_oneminus:
            FLAGS.pretrained_path += '_oneminus'
            FLAGS.exp_name += '_oneminus'
    if FLAGS.sep:
        FLAGS.pretrained_path += '_sep'
        FLAGS.exp_name += '_sep'
    if FLAGS.tag_tokens:
        FLAGS.pretrained_path += '_tag'
        FLAGS.exp_name += '_tag' + str(FLAGS.tag_loss_wt)
    if FLAGS.binarize_method == 'threshold':
        FLAGS.exp_name += '_thresh' + str(FLAGS.tag_threshold)
    elif FLAGS.binarize_method == 'summ_limit':
        FLAGS.exp_name += '_summ' + str(FLAGS.summ_limit)
    elif FLAGS.binarize_method == 'inst_limit':
        FLAGS.exp_name += '_inst' + str(FLAGS.inst_limit)
    else:
        raise Exception('Binarize method (%s) is not one of the choices.' % FLAGS.binarize_method)
    if FLAGS.only_tags:
        FLAGS.exp_name += '_onlytags'
    if FLAGS.ninst != -1:
        FLAGS.exp_name += '_ninst' + str(FLAGS.ninst)

    if FLAGS.resolver != 'stanford':
        FLAGS.data_root += '_' + FLAGS.resolver
    if FLAGS.triples_only:
        FLAGS.data_root += '_summinc_triples'
    if FLAGS.coref_dataset:
        FLAGS.data_root += '_fusions'
    if FLAGS.poc_dataset:
        FLAGS.data_root = os.path.expanduser('~') + '/data/tf_data/poc_fusions'


    bert_suffix = ''
    # if FLAGS.use_bert:
    #     if FLAGS.sentemb:
    #         FLAGS.exp_name += '_sentemb'
    #         bert_suffix += '_sentemb'
    #     if FLAGS.artemb:
    #         FLAGS.exp_name += '_artemb'
    #         bert_suffix += '_artemb'
    #     if FLAGS.plushidden:
    #         FLAGS.exp_name += '_plushidden'
    #         bert_suffix += '_plushidden'
    if FLAGS.tag_tokens:
        bert_suffix += '_tag' + str(FLAGS.tag_loss_wt)
    else:
        bert_suffix += '_tag' + '0.0'
    if FLAGS.upper_bound:
        FLAGS.exp_name = FLAGS.exp_name + '_upperbound'
        ssi_list = None     # this is if we are doing the upper bound evaluation (ssi_list comes straight from the groundtruth)
    else:
        if FLAGS.coref_dataset or FLAGS.poc_dataset:
            ssi_list = None
        else:
            my_log_dir = os.path.join(log_dir, '%s_%s_%s%s' % (FLAGS.dataset_name, extractor, FLAGS.singles_and_pairs, bert_suffix))
            print(util.bcolors.OKGREEN + "BERT path: " + my_log_dir + util.bcolors.ENDC)
            with open(os.path.join(my_log_dir, 'ssi.pkl'), 'rb') as f:
                ssi_list = pickle.load(f)
            FLAGS.ssi_data_path = my_log_dir
    if FLAGS.cnn_dm_pg:
        FLAGS.exp_name = FLAGS.exp_name + '_cnntrained'
    if FLAGS.websplit:
        FLAGS.exp_name = FLAGS.exp_name + '_websplittrained'
    if FLAGS.first_intact:
        FLAGS.exp_name = FLAGS.exp_name + '_firstintact'
    if FLAGS.upper_bound_ssi_only:
        FLAGS.exp_name = FLAGS.exp_name + '_upperboundssionly'


    print('Running statistics on %s' % FLAGS.exp_name)

    if FLAGS.dataset_name != "":
        FLAGS.data_path = os.path.join(FLAGS.data_root, FLAGS.dataset_name, FLAGS.dataset_split + '*')
    print(util.bcolors.WARNING + "Data path: " + FLAGS.data_path)
    if not os.path.exists(os.path.join(FLAGS.data_root, FLAGS.dataset_name)) or len(os.listdir(os.path.join(FLAGS.data_root, FLAGS.dataset_name))) == 0:
        print(('No TF example data found at %s' % os.path.join(FLAGS.data_root, FLAGS.dataset_name)))
        raise
        # convert_data.process_dataset(FLAGS.dataset_name)

    logging.set_verbosity(logging.INFO) # choose what level of logging you want
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.exp_name = FLAGS.exp_name if FLAGS.exp_name != '' else FLAGS.dataset_name
    FLAGS.actual_log_root = FLAGS.log_root
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

    print(util.bcolors.OKGREEN + "Experiment path: " + FLAGS.log_root + util.bcolors.ENDC)


    if FLAGS.dataset_name == 'duc_2004':
        vocab = Vocab(FLAGS.vocab_path + '_' + 'cnn_dm', FLAGS.vocab_size, add_sep=FLAGS.sep) # create a vocabulary
    else:
        vocab_datasets = [os.path.basename(file_path).split('vocab_')[1] for file_path in glob.glob(FLAGS.vocab_path + '_*')]
        original_dataset_name = [file_name for file_name in vocab_datasets if file_name in FLAGS.dataset_name]
        if len(original_dataset_name) > 1:
            raise Exception('Too many choices for vocab file')
        if len(original_dataset_name) < 1:
            raise Exception('No vocab file for dataset created. Run make_vocab.py --dataset_name=<my original dataset name>')
        original_dataset_name = original_dataset_name[0]
        FLAGS.original_dataset_name = original_dataset_name
        vocab = Vocab(FLAGS.vocab_path + '_' + original_dataset_name, FLAGS.vocab_size, add_sep=FLAGS.sep) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    # hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std',
    #                'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps',
    #                'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'lambdamart_input', 'pg_mmr', 'singles_and_pairs', 'skip_with_less_than_3',
    #                'ssi_data_path', 'word_imp_reg', 'imp_loss_wt']
    hparam_list = [item for item in list(FLAGS.flag_values_dict().keys()) if item != '?']
    hps_dict = {}
    for key,val in FLAGS.__flags.items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val.value # add it to the dict
    hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

    # tf.set_random_seed(113) # a seed value for randomness

    decode_model_hps = hps._replace(
        max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(FLAGS.data_root, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + FLAGS.dataset_split + '*'))

    # total = len(source_files) * 1000 if 'cnn' in dataset_articles or 'xsum' in dataset_articles else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + FLAGS.dataset_split + '*', True, False,
                                               should_check_valid=False)
    # batcher = Batcher(None, vocab, hps, single_pass=FLAGS.single_pass)
    if FLAGS.unilm_decoding:
        model = None
    else:
        model = SummarizationModel(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(model, None, vocab)
    decoder.decode_iteratively(example_generator, num_test_examples, names_to_types, ssi_list, hps)

    # num_outside = []
    # for example_idx, example in enumerate(tqdm(example_generator, total=total)):
    #     raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs = util.unpack_tf_example(
    #         example, names_to_types)
    #     article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
    #     cur_token_idx = 0
    #     for sent_idx, sent_tokens in enumerate(article_sent_tokens):
    #         for token in sent_tokens:
    #             cur_token_idx += 1
    #             if cur_token_idx >= 400:
    #                 sent_idx_at_400 = sent_idx
    #                 break
    #         if cur_token_idx >= 400:
    #             break
    #
    #     my_num_outside = 0
    #     for ssi in groundtruth_similar_source_indices_list:
    #         for source_idx in ssi:
    #             if source_idx >= sent_idx_at_400:
    #                 my_num_outside += 1
    #     num_outside.append(my_num_outside)
    # print "num_outside = %d" % np.mean(num_outside)


    a=0

if __name__ == '__main__':
    app.run(main)

























