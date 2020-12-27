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
import json


FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')
if 'coref' not in flags.FLAGS:
    flags.DEFINE_boolean('coref', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
if 'poc_dataset' not in flags.FLAGS:
    flags.DEFINE_boolean('poc_dataset', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_string("resolver", "spacy", "Which method to use for turning token tag probabilities into binary tags. Can be one of {threshold, summ_limit, inst_limit}.")

FLAGS(sys.argv)

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens'
data_dir += '_' + FLAGS.resolver
if FLAGS.coref:
    if FLAGS.poc_dataset:
        data_dir = os.path.expanduser('~') + '/data/tf_data/poc_fusions'
    else:
        data_dir += '_fusions'
if FLAGS.coref:
    if FLAGS.poc_dataset:
        names_to_types = [('raw_article_sents', 'string_list'), ('summary_text', 'string'), ('coref_chains', 'delimited_list_of_list_of_lists')]
    else:
        names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'),
                      ('article_lcs_paths_list', 'delimited_list_of_list_of_lists'), ('coref_chains', 'delimited_list_of_list_of_lists'), ('coref_representatives', 'string_list')]
else:
    names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'),
                  ('doc_indices', 'delimited_list'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]
min_matched_tokens = 1
np.random.seed(123)
chronological_ssi = True

def flatten_coref_chains(coref_chains, raw_article_sents, ssi):
    article_sent1 = raw_article_sents[ssi[0]]
    article_sent2 = raw_article_sents[ssi[1]]
    num_tokens_sent1 = len(article_sent1.split(' '))
    num_tokens_sent2 = len(article_sent2.split(' '))
    flat_coref_chains = []
    for chain in coref_chains:
        flat_chain = []
        for mention in chain:
            if mention[0] == 1:
                flat_mention = (num_tokens_sent1 + mention[1], num_tokens_sent1 + mention[2])
            else:
                flat_mention = (mention[1], mention[2])
            flat_chain.append(flat_mention)
        flat_coref_chains.append(flat_chain)
    return flat_coref_chains


def get_string_bert_example(output_article_text, raw_article_sents, groundtruth_summ_sent, ssi, coref_chains, example_idx, inst_id, sent2_start):
    instance = [
        output_article_text,
        groundtruth_summ_sent,
        str(example_idx),
        str(inst_id),
        # ' '.join([str(i) for i in ssi]),
        str(sent2_start)
    ]
    if coref_chains is not None:
        flat_coref_chains = flatten_coref_chains(coref_chains, raw_article_sents, ssi)
        coref_chains_dict = {}
        for chain_idx, chain in enumerate(flat_coref_chains):
            coref_chains_dict['coref_chain_%d' % chain_idx] = []
            for mention in chain:
                coref_chains_dict['coref_chain_%d' % chain_idx].append({'start': mention[0], 'end': mention[1]})
        coref_chains_str = json.dumps(coref_chains_dict)
        instance.append(coref_chains_str)
    return '\t'.join(instance) + '\n'


def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.singles_and_pairs == 'singles':
        FLAGS.sentence_limit = 1
    else:
        FLAGS.sentence_limit = 2


    if FLAGS.dataset_name == 'all':
        dataset_names = ['cnn_dm', 'xsum', 'duc_2004']
    else:
        dataset_names = [FLAGS.dataset_name]

    for dataset_name in dataset_names:
        FLAGS.dataset_name = dataset_name


        source_dir = os.path.join(data_dir, dataset_name)

        if FLAGS.dataset_split == 'all':
            if FLAGS.poc_dataset:
                dataset_splits = ['test']
            else:
                dataset_splits = ['test', 'val', 'train']
        else:
            dataset_splits = [FLAGS.dataset_split]


        for dataset_split in dataset_splits:
            source_dataset_split = dataset_split

            source_files = sorted(glob.glob(source_dir + '/' + source_dataset_split + '*'))

            total = len(source_files) * 1000
            example_generator = data.example_generator(source_dir + '/' + source_dataset_split + '*', True, False,
                                                       should_check_valid=False)

            out_dir = os.path.join('data', 'bert', dataset_name, FLAGS.singles_and_pairs, 'input_decoding')
            if FLAGS.coref:
                if FLAGS.poc_dataset:
                    out_dir += '_pocd'
                else:
                    out_dir += '_crd'
            util.create_dirs(out_dir)

            writer = open(os.path.join(out_dir, dataset_split) + '.tsv', 'wb')
            header_list = [
                'article_sents',
                'summary_sent',
                'article_id',
                'instance_id',
                # 'ssi',
                'sent_2_start_token_idx'
            ]
            if FLAGS.coref:
                header_list.append('coref_chains')
            writer.write(('\t'.join(header_list) + '\n').encode())
            inst_id = 0
            for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                if FLAGS.coref:
                    if FLAGS.poc_dataset:
                        raw_article_sents, groundtruth_summary_text, coref_chains = util.unpack_tf_example(
                            example, names_to_types)
                        groundtruth_similar_source_indices_list = [[0, 1]]
                    else:
                        raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, article_lcs_paths_list, coref_chains, coref_representatives = util.unpack_tf_example(
                            example, names_to_types)
                    doc_indices = None
                else:
                    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices, article_lcs_paths_list = util.unpack_tf_example(
                        example, names_to_types)
                    coref_chains = None
                article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
                groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
                if doc_indices is None or (dataset_name != 'duc_2004' and len(doc_indices) != len(util.flatten_list_of_lists(article_sent_tokens))):
                    doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
                doc_indices = [int(doc_idx) for doc_idx in doc_indices]
                rel_sent_indices, _, _ = util.get_rel_sent_indices(doc_indices, article_sent_tokens)
                similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)
                # print doc_indices, rel_sent_indices



                positives = [ssi for ssi in similar_source_indices_list]
                positive_sents = list(set(util.flatten_list_of_lists(positives)))

                for ssi_idx, ssi in enumerate(similar_source_indices_list):
                    output_article_text = ' '.join(util.reorder(raw_article_sents, ssi))
                    if len(ssi) == 0:
                        continue
                    if chronological_ssi and len(ssi) >= 2:
                        if ssi[0] > ssi[1]:
                            ssi = (min(ssi), max(ssi))
                    is_pair = len(ssi) == 2
                    sent2_start = len(raw_article_sents[ssi[0]].split())
                    writer.write(get_string_bert_example(output_article_text, raw_article_sents, groundtruth_summ_sents[0][ssi_idx], ssi, coref_chains, example_idx, inst_id, sent2_start).encode())
                    inst_id += 1




if __name__ == '__main__':
    app.run(main)



