import ssi_functions

import sys
import nltk
import numpy as np
import struct
import os
import glob
import convert_data
from absl import flags
from absl import app
import pickle
import util
from data import Vocab
from difflib import SequenceMatcher
import itertools
from tqdm import tqdm
import importance_features
import data
import json
import copy
# from pycorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('http://localhost:9000')
#
# path = os.path.expanduser('~') + "/data/discourse/newsroom/data/test-stats.jsonl"
# data = []
#
# with open(path) as f:
#     for ln in f:
#         obj = json.loads(ln)
#         data.append(obj)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

FLAGS = flags.FLAGS


data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref'
log_dir = 'logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120
np.random.seed(123)

threshold = 0.9
default_exp_name = 'duc_2004_reservoir_lambda_0.6_mute_7_tfidf'

html_dir = 'data/highlight'
ssi_dir = 'data/ssi'
kaiqiang_dir = 'data/kaiqiang_single_sent_data'
lambdamart_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'

def get_summary_text(summary_file):
    with open(summary_file) as f:
        summary_text = f.read()
    return summary_text

def get_summary_from_example(e):
    summary_texts = []
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract.decode())  # the abstracts texts was saved under the key 'abstract' in the data files
    all_abstract_sentences = [[sent.strip() for sent in data.abstract2sents(
        abstract)] for abstract in summary_texts]
    summary_text = '\n'.join(all_abstract_sentences[0])
    all_summary_texts = ['\n'.join(abs_sents) for abs_sents in all_abstract_sentences]
    return summary_text, all_summary_texts

def split_into_tokens(text):
    tokens = text.split()
    tokens = [t for t in tokens if t != '<s>' and t != '</s>']
    return tokens

def split_into_sent_tokens(text):
    sent_tokens = [[t for t in tokens.strip().split() if t != '<s>' and t != '</s>'] for tokens in text.strip().split('\n')]
    return sent_tokens

def cluster_similar_source_sents(article_sent_tokens, similar_source_indices, vocab, threshold):
    chosen_article_sents = [sent for i, sent in enumerate(article_sent_tokens) if i in similar_source_indices]
    temp_similarity_matrix = util.rouge_l_similarity_matrix(chosen_article_sents,
                                         chosen_article_sents, vocab, 'f1')
    similarity_matrix = np.zeros([len(article_sent_tokens), len(article_sent_tokens)], dtype=float)
    for row_idx in range(len(temp_similarity_matrix)):
        for col_idx in range(len(temp_similarity_matrix)):
            similarity_matrix[similar_source_indices[row_idx], similar_source_indices[col_idx]] = temp_similarity_matrix[row_idx, col_idx]

    groups = [[similar_source_indices[0]]]
    for sent_idx in similar_source_indices[1:]:
        found_group = False
        for group in groups:
            for group_member in group:
                similarity = similarity_matrix[sent_idx, group_member]
                if similarity >= threshold:
                    found_group = True
                    group.append(sent_idx)
                    break
            if found_group:
                break
        if not found_group:
            groups.append([sent_idx])
    return groups

def get_shortest_distance(indices1, indices2, relative_to_article, rel_sent_positions):
    if relative_to_article:
        indices1 = [rel_sent_positions[idx] for idx in indices1]
        indices2 = [rel_sent_positions[idx] for idx in indices2]
    pairs = list(itertools.product(indices1, indices2))
    min_dist = min([abs(x - y) for x,y in pairs])
    return min_dist

def get_merge_example(similar_source_indices, article_sent_tokens, summ_sent, corefs, article_lcs_paths):
    # restricted_source_indices = []
    # for source_indices_idx, source_indices in enumerate(similar_source_indices):
    #     if source_indices_idx >= FLAGS.sentence_limit:
    #         break
    #     restricted_source_indices.append(source_indices[0])
    if FLAGS.chronological and len(similar_source_indices) > 1:
        if similar_source_indices[0] > similar_source_indices[1]:
            similar_source_indices = (min(similar_source_indices), max(similar_source_indices))
            article_lcs_paths = (article_lcs_paths[1], article_lcs_paths[0])
    merged_example_sentences = [' '.join(sent) for sent in util.reorder(article_sent_tokens, similar_source_indices)]
    merged_example_article_text = ' '.join(merged_example_sentences)
    merged_example_abstracts = [[' '.join(summ_sent)]]
    merge_example = convert_data.make_example(merged_example_article_text, merged_example_abstracts, None, merged_example_sentences, corefs, article_lcs_paths)
    return merge_example

def get_kaiqiang_article_abstract(similar_source_indices, raw_article_sents, summ_sent):
    source_idx = similar_source_indices[0][0]
    article_text = raw_article_sents[source_idx]
    abstract_text = ' '.join(summ_sent)
    return article_text, abstract_text

def get_single_sent_features(similar_source_indices, sent_term_matrix, doc_vector, article_sent_tokens):
    sent_idx = similar_source_indices[0]
    doc_similarity = util.cosine_similarity(sent_term_matrix[sent_idx], doc_vector)
    sent_len = len(article_sent_tokens[sent_idx])
    return sent_idx, doc_similarity, sent_len


def main(unused_argv):

    print('Running statistics on %s' % FLAGS.exp_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.all_actions:
        FLAGS.sent_dataset = True
        FLAGS.ssi_dataset = True
        FLAGS.print_output = True
        FLAGS.highlight = True

    original_dataset_name = 'xsum' if 'xsum' in FLAGS.dataset_name else 'cnn_dm' if ('cnn_dm' in FLAGS.dataset_name or 'duc_2004' in FLAGS.dataset_name) else ''
    vocab = Vocab(FLAGS.vocab_path + '_' + original_dataset_name, FLAGS.vocab_size) # create a vocabulary

    source_dir = os.path.join(data_dir, FLAGS.dataset_name)
    util.create_dirs(html_dir)

    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for dataset_split in dataset_splits:
        source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
        if FLAGS.exp_name == 'reference':
            # summary_dir = log_dir + default_exp_name + '/decode_test_' + str(max_enc_steps) + \
            #                 'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/reference'
            # summary_files = sorted(glob.glob(summary_dir + '/*_reference.A.txt'))
            summary_dir = source_dir
            summary_files = source_files
        else:
            if FLAGS.exp_name == 'cnn_dm':
                summary_dir = log_dir + FLAGS.exp_name + '/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded'
            else:
                ckpt_folder = util.find_largest_ckpt_folder(log_dir + FLAGS.exp_name)
                summary_dir = log_dir + FLAGS.exp_name + '/' + ckpt_folder + '/decoded'
                # summary_dir = log_dir + FLAGS.exp_name + '/decode_test_' + str(max_enc_steps) + \
                #             'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-238410/decoded'
            summary_files = sorted(glob.glob(summary_dir + '/*'))
        if len(summary_files) == 0:
            raise Exception('No files found in %s' % summary_dir)
        example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, is_original=True)
        pros = {'annotators': 'dcoref', 'outputFormat': 'json', 'timeout': '5000000'}
        all_merge_examples = []
        num_extracted_list = []
        distances = []
        relative_distances = []
        html_str = ''
        extracted_sents_in_article_html = ''
        name = FLAGS.dataset_name + '_' + FLAGS.exp_name
        if FLAGS.coreference_replacement:
            name += '_coref'
        highlight_file_name = os.path.join(html_dir, FLAGS.dataset_name + '_' + FLAGS.exp_name)
        if FLAGS.consider_stopwords:
            highlight_file_name += '_stopwords'
        if FLAGS.highlight:
            extracted_sents_in_article_html_file = open(highlight_file_name + '_extracted_sents.html', 'wb')
        if FLAGS.kaiqiang:
            kaiqiang_article_texts = []
            kaiqiang_abstract_texts = []
            util.create_dirs(kaiqiang_dir)
            kaiqiang_article_file = open(os.path.join(kaiqiang_dir, FLAGS.dataset_name + '_' + dataset_split + '_' + str(FLAGS.min_matched_tokens) + '_articles.txt'), 'wb')
            kaiqiang_abstract_file = open(os.path.join(kaiqiang_dir, FLAGS.dataset_name + '_' + dataset_split + '_' + str(FLAGS.min_matched_tokens)  + '_abstracts.txt'), 'wb')
        if FLAGS.ssi_dataset:
            if FLAGS.tag_tokens:
                with_coref_and_ssi_dir = lambdamart_dir + '_and_tag_tokens'
            else:
                with_coref_and_ssi_dir = lambdamart_dir
            lambdamart_out_dir = os.path.join(with_coref_and_ssi_dir, FLAGS.dataset_name)
            if FLAGS.sentence_limit == 1:
                lambdamart_out_dir += '_singles'
            if FLAGS.consider_stopwords:
                lambdamart_out_dir += '_stopwords'
            lambdamart_out_full_dir = os.path.join(lambdamart_out_dir, 'all')
            util.create_dirs(lambdamart_out_full_dir)
            lambdamart_writer = open(os.path.join(lambdamart_out_full_dir, dataset_split + '.bin'), 'wb')

        simple_similar_source_indices_list_plus_empty = []
        example_idx = -1
        instance_idx = 0
        total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
        random_choices = None
        if FLAGS.randomize:
            if FLAGS.dataset_name == 'cnn_dm':
                list_order = np.random.permutation(11490)
                random_choices = list_order[:FLAGS.num_instances]
        for example in tqdm(example_generator, total=total):
            example_idx += 1
            if FLAGS.num_instances != -1 and instance_idx >= FLAGS.num_instances:
                break
            if random_choices is not None and example_idx not in random_choices:
                continue
        # for file_idx in tqdm(range(len(source_files))):
        #     example = get_tf_example(source_files[file_idx])
            article_text = example.features.feature['article'].bytes_list.value[0].decode().lower()
            if FLAGS.exp_name == 'reference':
                summary_text, all_summary_texts = get_summary_from_example(example)
            else:
                summary_text = get_summary_text(summary_files[example_idx])
            article_tokens = split_into_tokens(article_text)
            if 'raw_article_sents' in example.features.feature and len(example.features.feature['raw_article_sents'].bytes_list.value) > 0:
                raw_article_sents = example.features.feature['raw_article_sents'].bytes_list.value

                raw_article_sents = [sent.decode() for sent in raw_article_sents if sent.decode().strip() != '']
                article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
            else:
                # article_text = util.to_unicode(article_text)

                # sent_pros = {'annotators': 'ssplit', 'outputFormat': 'json', 'timeout': '5000000'}
                # sents_result_dict = nlp.annotate(str(article_text), properties=sent_pros)
                # article_sent_tokens = [[token['word'] for token in sent['tokens']] for sent in sents_result_dict['sentences']]

                raw_article_sents = nltk.tokenize.sent_tokenize(article_text)
                article_sent_tokens = [util.process_sent(sent) for sent in raw_article_sents]
            if FLAGS.top_n_sents != -1:
                article_sent_tokens = article_sent_tokens[:FLAGS.top_n_sents]
                raw_article_sents = raw_article_sents[:FLAGS.top_n_sents]
            article_sents = [' '.join(sent) for sent in article_sent_tokens]
            try:
                article_tokens_string = str(' '.join(article_sents))
            except:
                try:
                    article_tokens_string = str(' '.join([sent.decode('latin-1') for sent in article_sents]))
                except:
                    raise


            if len(article_sent_tokens) == 0:
                continue

            summary_sent_tokens = split_into_sent_tokens(summary_text)
            if 'doc_indices' in example.features.feature and len(example.features.feature['doc_indices'].bytes_list.value) > 0:
                doc_indices_str = example.features.feature['doc_indices'].bytes_list.value[0].decode()
                if '1' in doc_indices_str:
                    doc_indices = [int(x) for x in doc_indices_str.strip().split()]
                    rel_sent_positions = importance_features.get_sent_indices(article_sent_tokens, doc_indices)
                else:
                    num_tokens_total = sum([len(sent) for sent in article_sent_tokens])
                    rel_sent_positions = list(range(len(raw_article_sents)))
                    doc_indices = [0] * num_tokens_total

            else:
                rel_sent_positions = None
                doc_indices = None
                doc_indices_str = None
            if 'corefs' in example.features.feature and len(
                    example.features.feature['corefs'].bytes_list.value) > 0:
                corefs_str = example.features.feature['corefs'].bytes_list.value[0]
                corefs = json.loads(corefs_str)
            # summary_sent_tokens = limit_to_n_tokens(summary_sent_tokens, 100)

            similar_source_indices_list_plus_empty = []

            simple_similar_source_indices, lcs_paths_list, article_lcs_paths_list, smooth_article_paths_list =  ssi_functions.get_simple_source_indices_list(
                summary_sent_tokens, article_sent_tokens, vocab, FLAGS.sentence_limit, FLAGS.min_matched_tokens, not FLAGS.consider_stopwords, lemmatize=FLAGS.lemmatize,
                multiple_ssi=FLAGS.multiple_ssi)

            article_paths_parameter = article_lcs_paths_list if FLAGS.tag_tokens else None
            article_paths_parameter = smooth_article_paths_list if FLAGS.smart_tags else article_paths_parameter
            restricted_source_indices = util.enforce_sentence_limit(simple_similar_source_indices, FLAGS.sentence_limit)
            for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
                if FLAGS.sent_dataset:
                    if len(restricted_source_indices[summ_sent_idx]) == 0:
                        continue
                    merge_example = get_merge_example(restricted_source_indices[summ_sent_idx], article_sent_tokens, summ_sent, corefs, article_paths_parameter[summ_sent_idx])
                    all_merge_examples.append(merge_example)

            simple_similar_source_indices_list_plus_empty.append(simple_similar_source_indices)
            if FLAGS.ssi_dataset:
                summary_text_to_save = [s for s in all_summary_texts] if FLAGS.dataset_name == 'duc_2004' else summary_text
                new_tf_example = util.make_tf_example(simple_similar_source_indices, raw_article_sents, summary_text_to_save, corefs_str, doc_indices_str, article_paths_parameter, None)
                util.write_tf_example(lambdamart_writer, new_tf_example)


            if FLAGS.highlight:
                highlight_article_lcs_paths_list = smooth_article_paths_list if FLAGS.smart_tags else article_lcs_paths_list
                # simple_ssi_plus_empty = [ [s[0] for s in sim_source_ind] for sim_source_ind in simple_similar_source_indices]
                extracted_sents_in_article_html = ssi_functions.html_highlight_sents_in_article(summary_sent_tokens, simple_similar_source_indices,
                                                                                  article_sent_tokens, doc_indices,
                                                                                  lcs_paths_list, highlight_article_lcs_paths_list)
                extracted_sents_in_article_html_file.write(extracted_sents_in_article_html.encode())
            a=0

            instance_idx += 1


        if FLAGS.ssi_dataset:
            lambdamart_writer.close()
            if FLAGS.dataset_name == 'cnn_dm' or FLAGS.dataset_name == 'newsroom' or FLAGS.dataset_name == 'xsum':
                chunk_size = 1000
            else:
                chunk_size = 1
            util.chunk_file(dataset_split, lambdamart_out_full_dir, lambdamart_out_dir, chunk_size=chunk_size)

        if FLAGS.sent_dataset:
            with_coref_dir = data_dir + '_and_tag_tokens' if FLAGS.tag_tokens else data_dir
            out_dir = os.path.join(with_coref_dir, FLAGS.dataset_name + '_sent')
            if FLAGS.sentence_limit == 1:
                out_dir += '_singles'
            if FLAGS.consider_stopwords:
                out_dir += '_stopwords'
            if FLAGS.coreference_replacement:
                out_dir += '_coref'
            if FLAGS.top_n_sents != -1:
                out_dir += '_n=' + str(FLAGS.top_n_sents)
            util.create_dirs(out_dir)
            convert_data.write_with_generator(iter(all_merge_examples), len(all_merge_examples), out_dir, dataset_split)

        if FLAGS.print_output:
            # html_str = FLAGS.dataset + ' | ' + FLAGS.exp_name + '<br><br><br>' + html_str
            # save_fusions_to_file(html_str)
            ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name)
            if FLAGS.consider_stopwords:
                ssi_path += '_stopwords'
            util.create_dirs(ssi_path)
            if FLAGS.dataset_name == 'duc_2004' and FLAGS.abstract_idx != 0:
                abstract_idx_str = '_%d' % FLAGS.abstract_idx
            else:
                abstract_idx_str = ''
            with open(os.path.join(ssi_path, dataset_split + '_ssi' + abstract_idx_str + '.pkl'), 'wb') as f:
                pickle.dump(simple_similar_source_indices_list_plus_empty, f)

        if FLAGS.kaiqiang:
            # kaiqiang_article_file.write('\n'.join(kaiqiang_article_texts))
            # kaiqiang_abstract_file.write('\n'.join(kaiqiang_abstract_texts))
            kaiqiang_article_file.close()
            kaiqiang_abstract_file.close()
        if FLAGS.highlight:
            extracted_sents_in_article_html_file.close()
        a=0


if __name__ == '__main__':
    flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
                               ' If you want to run on human summaries, then enter "reference".')
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
    flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
    flags.DEFINE_integer('num_instances', -1, 'Number of instances to run for before stopping. Use -1 to run on all instances.')
    flags.DEFINE_string('vocab_path', 'logs/vocab', 'Path expression to text vocabulary file.')
    flags.DEFINE_boolean('only_rouge_l', False, 'Whether to use only R-L in calculating similarity or whether to average over R-1, R-2, and R-L.')
    flags.DEFINE_boolean('coreference_replacement', False, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('kaiqiang', False, 'Whether to save the single sentences as a dataset for Kaiqiang.')
    flags.DEFINE_integer('top_n_sents', -1, 'Number of sentences to take from the beginning of the article. Use -1 to run on entire article.')
    flags.DEFINE_integer('min_matched_tokens', 2, 'Number of tokens required that still counts a source sentence as matching a summary sentence.')
    flags.DEFINE_integer('abstract_idx', 0, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('consider_stopwords', False, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('print_output', False, 'Whether to print and save the merged sentences and statistics.')
    flags.DEFINE_boolean('highlight', False, 'Whether to save an html file that shows the selected sentences as highlighted in the article.')
    flags.DEFINE_boolean('sent_dataset', False, 'Whether to save the merged sentences as a dataset.')
    flags.DEFINE_boolean('ssi_dataset', False, 'Whether to save features as a dataset that will be used to predict which sentences should be merged, using the LambdaMART system.')
    flags.DEFINE_boolean('all_actions', False, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('lemmatize', True, 'Which human abstract to process on. Only applies to duc_2004.')
    flags.DEFINE_boolean('multiple_ssi', False, 'Allow multiple singles are pairs to be chosen for each summary sentence, rather than just the top similar sentence.')
    flags.DEFINE_boolean('chronological', True, 'Whether to make sent_dataset chronological for source indices. Does not apply to ssi_dataset.')
    flags.DEFINE_boolean('randomize', False, 'Whether to make sent_dataset chronological for source indices. Does not apply to ssi_dataset.')
    flags.DEFINE_boolean('tag_tokens', True, 'Whether to add token-level tags, representing whether this token is copied from the source to the summary.')
    flags.DEFINE_boolean('smart_tags', True, 'Whether to add token-level tags, representing whether this token is copied from the source to the summary.')

    app.run(main)

















