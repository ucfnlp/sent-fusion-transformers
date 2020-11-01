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

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)


import convert_data
import lambdamart_scores_to_summaries
import preprocess_for_lambdamart_no_flags

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]
min_matched_tokens = 1

def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    source_dir = os.path.join(data_dir, FLAGS.dataset_name)
    source_files = sorted(glob.glob(source_dir + '/' + FLAGS.dataset_split + '*'))

    total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + FLAGS.dataset_split + '*', True, False,
                                               should_check_valid=False)

    for example_idx, example in enumerate(tqdm(example_generator, total=total)):
        raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(
            example, names_to_types)
        article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
        groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
        if doc_indices is None:
            doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
        doc_indices = [int(doc_idx) for doc_idx in doc_indices]
        rel_sent_indices, _, _ = preprocess_for_lambdamart_no_flags.get_rel_sent_indices(doc_indices, article_sent_tokens)
        groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)




if __name__ == '__main__':
    app.run(main)



