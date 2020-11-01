
from data import Vocab
import nltk

import rouge_functions
import itertools
import os
from tqdm import tqdm
import numpy as np
from absl import flags
from absl import app
import pickle
import util
import sys
import glob
import data
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from count_merged import get_simple_source_indices_list

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'all', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'summarizer' not in flags.FLAGS:
    flags.DEFINE_string('summarizer', 'all', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)


import convert_data

log_root = 'logs'
data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]
min_matched_tokens = 1

summarizers = {
    'lexrank': LexRankSummarizer,
    'kl': KLSummarizer,
    'sumbasic': SumBasicSummarizer
}
datasets = ['duc_2004', 'xsum', 'cnn_dm']

def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.summarizer == 'all':
        summary_methods = list(summarizers.keys())
    else:
        summary_methods = [FLAGS.summarizer]
    if FLAGS.dataset_name == 'all':
        dataset_names = datasets
    else:
        dataset_names = [FLAGS.dataset_name]

    sheets_strs = []
    for summary_method in summary_methods:
        summary_fn = summarizers[summary_method]
        for dataset_name in dataset_names:
            FLAGS.dataset_name = dataset_name

            original_dataset_name = 'xsum' if 'xsum' in dataset_name else 'cnn_dm' if 'cnn_dm' in dataset_name or 'duc_2004' in dataset_name else ''
            vocab = Vocab('logs/vocab' + '_' + original_dataset_name, 50000)  # create a vocabulary

            source_dir = os.path.join(data_dir, dataset_name)
            source_files = sorted(glob.glob(source_dir + '/' + FLAGS.dataset_split + '*'))

            total = len(source_files) * 1000 if ('cnn' in dataset_name or 'newsroom' in dataset_name or 'xsum' in dataset_name) else len(source_files)
            example_generator = data.example_generator(source_dir + '/' + FLAGS.dataset_split + '*', True, False,
                                                       should_check_valid=False)

            if dataset_name == 'duc_2004':
                abs_source_dir = os.path.join(os.path.expanduser('~') + '/data/tf_data/with_coref', dataset_name)
                abs_example_generator = data.example_generator(abs_source_dir + '/' + FLAGS.dataset_split + '*', True, False,
                                                           should_check_valid=False)
                abs_names_to_types = [('abstract', 'string_list')]

            triplet_ssi_list = []
            for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(
                    example, names_to_types)
                if dataset_name == 'duc_2004':
                    abs_example = next(abs_example_generator)
                    groundtruth_summary_texts = util.unpack_tf_example(abs_example, abs_names_to_types)
                    groundtruth_summary_texts = groundtruth_summary_texts[0]
                    groundtruth_summ_sents_list = [[sent.strip() for sent in data.abstract2sents(
                        abstract)] for abstract in groundtruth_summary_texts]

                else:
                    groundtruth_summary_texts = [groundtruth_summary_text]
                    groundtruth_summ_sents_list = []
                    for groundtruth_summary_text in groundtruth_summary_texts:
                        groundtruth_summ_sents = [sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]
                        groundtruth_summ_sents_list.append(groundtruth_summ_sents)
                article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
                if doc_indices is None:
                    doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
                doc_indices = [int(doc_idx) for doc_idx in doc_indices]
                groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)

                log_dir = os.path.join(log_root, dataset_name + '_' + summary_method)
                dec_dir = os.path.join(log_dir, 'decoded')
                ref_dir = os.path.join(log_dir, 'reference')
                util.create_dirs(dec_dir)
                util.create_dirs(ref_dir)

                parser = PlaintextParser.from_string(' '.join(raw_article_sents), Tokenizer("english"))
                summarizer = summary_fn()

                summary = summarizer(parser.document, 5) #Summarize the document with 5 sentences
                summary = [str(sentence) for sentence in summary]

                summary_tokenized = []
                for sent in summary:
                    summary_tokenized.append(sent.lower())

                rouge_functions.write_for_rouge(groundtruth_summ_sents_list, summary_tokenized, example_idx, ref_dir, dec_dir, log=False)

                decoded_sent_tokens = [sent.split() for sent in summary_tokenized]
                sentence_limit = 2
                sys_ssi_list, _, _ = get_simple_source_indices_list(decoded_sent_tokens, article_sent_tokens, vocab, sentence_limit,
                                               min_matched_tokens)
                triplet_ssi_list.append((groundtruth_similar_source_indices_list, sys_ssi_list, -1))


            print('Evaluating Lambdamart model F1 score...')
            suffix = util.all_sent_selection_eval(triplet_ssi_list)
            print(suffix)

            results_dict = rouge_functions.rouge_eval(ref_dir, dec_dir)
            print(("Results_dict: ", results_dict))
            sheets_str = rouge_functions.rouge_log(results_dict, log_dir, suffix=suffix)
            sheets_strs.append(dataset_name + '_' + summary_method + '\n' + sheets_str)

    for sheets_str in sheets_strs:
        print(sheets_str + '\n')




if __name__ == '__main__':
    app.run(main)



