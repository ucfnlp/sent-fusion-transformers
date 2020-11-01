# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications made 2018 by Logan Lebanoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""
import math

# import tensorflow as tf
import time
import os
import numpy as np
from absl import flags
import itertools
# import data
from absl import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
import inspect, re
import string
import struct
import rouge_functions
import json
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English
import sys
import nltk
import shutil
import unicodedata
# import backtrace
# from tensorflow.core.example import example_pb2

# backtrace.hook(
#     reverse=False,
#     align=False,
#     strip_path=False,
#     enable_on_envvar_only=False,
#     on_tty=False,
#     conservative=False,
#     styles={})

if sys.version_info >= (3, 0):
    python_version = 3
else:
    python_version = 2

nlp = English()
try:
    nlp2 = spacy.load('en', disable=['parser', 'ner'])
except:
    nlp2 = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
FLAGS = flags.FLAGS

stop_words = set(stopwords.words('english'))
CHUNK_SIZE = 1000

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
            if FLAGS.use_pretrained:
                my_ckpt_dir = os.path.join(FLAGS.pretrained_path, 'train')
            else:
                my_ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(my_ckpt_dir, latest_filename=latest_filename)
            print (bcolors.OKGREEN + 'Loading checkpoint %s' % ckpt_state.model_checkpoint_path + bcolors.ENDC)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", my_ckpt_dir, 10)
            time.sleep(10)
            raise

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def flatten_list_of_lists_3d(list_of_list_of_lists):
    return list(itertools.chain.from_iterable([itertools.chain.from_iterable(list_of_lists) for list_of_lists in list_of_list_of_lists]))

def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l.
    """
    chunk_list = []
    for i in range(0, len(chunkable), n):
        chunk_list.append( chunkable[i:i+n])
    return chunk_list

def is_list_type(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

def get_first_item(lst):
    if not is_list_type(lst):
        return lst
    for item in lst:
        result = get_first_item(item)
        if result is not None:
            return result
    return None

def remove_period_ids(lst, vocab):
    first_item = get_first_item(lst)
    if first_item is None:
        return lst
    if vocab is not None and type(first_item) == int:
        period = vocab.word2id(data.PERIOD)
    else:
        period = '.'

    if is_list_type(lst[0]):
        return [[item for item in inner_list if item != period] for inner_list in lst]
    else:
        return [item for item in lst if item != period]

def to_unicode(text):
    try:
        text = str(text, errors='replace')
    except TypeError:
        return text
    return text

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

def calc_ROUGE_L_score(candidate, reference, metric='f1'):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    beta = 1.2
    prec = []
    rec = []

    if len(reference) == 0:
        return 0.

    if type(reference[0]) is not list:
        reference = [reference]

    for ref in reference:
        # compute the longest common subsequence
        lcs = my_lcs(ref, candidate)
        try:
            prec.append(lcs / float(len(candidate)))
            rec.append(lcs / float(len(ref)))
        except:
            print('Candidate', candidate)
            print('Reference', ref)
            raise


    prec_max = max(prec)
    rec_max = max(rec)

    if metric == 'f1':
        if (prec_max != 0 and rec_max != 0):
            score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
        else:
            score = 0.0
    elif metric == 'precision':
        score = prec_max
    elif metric == 'recall':
        score = rec_max
    else:
        raise Exception('Invalid metric argument: %s. Must be one of {f1,precision,recall}.' % metric)
    return score

def create_token_to_indices(lst):
    token_to_indices = {}
    for token_idx, token in enumerate(lst):
        if token in token_to_indices:
            token_to_indices[token].append(token_idx)
        else:
            token_to_indices[token] = [token_idx]
    return token_to_indices

def matching_unigrams(summ_sent, article_sent, should_remove_stop_words=False, should_remove_punctuation=False):
    if should_remove_stop_words:
        summ_sent = remove_stopwords_punctuation(summ_sent)
        article_sent = remove_stopwords_punctuation(article_sent)
    matches = []
    summ_indices = []
    article_indices = []
    summ_token_to_indices = create_token_to_indices(summ_sent)
    article_token_to_indices = create_token_to_indices(article_sent)
    for token in list(summ_token_to_indices.keys()):
        if token in article_token_to_indices:
            summ_indices.extend(summ_token_to_indices[token])
            article_indices.extend(article_token_to_indices[token])
            matches.extend([token] * len(summ_token_to_indices[token]))
    summ_indices = sorted(summ_indices)
    article_indices = sorted(article_indices)
    return matches, (summ_indices, article_indices)

def is_punctuation(word):
    is_punctuation = [ch in string.punctuation for ch in word]
    if all(is_punctuation):
        return True
    return False

def is_stopword_punctuation(word):
    if word in stop_words or word in ('<s>', '</s>'):
        return True
    is_punctuation = [ch in string.punctuation for ch in word]
    if all(is_punctuation):
        return True
    return False

def is_content_word(word):
    return not is_stopword_punctuation(word)

def is_stopword(word):
    if word in stop_words:
        return True
    return False

def is_quotation_mark(word):
    if word in ["``", "''", "`", "'"]:
        return True
    return False

def is_start_stop_symbol(word):
    if word in ('<s>', '</s>'):
        return True
    return False

def remove_start_stop_symbol(sent):
    new_sent = [token for token in sent if not is_start_stop_symbol(token)]
    return new_sent

def remove_punctuation(sent):
    new_sent = [token for token in sent if not is_punctuation(token)]
    return new_sent

def remove_stopwords(sent):
    new_sent = [token for token in sent if not is_stopword(token)]
    return new_sent

def remove_stopwords_punctuation(sent):
    try:
        new_sent = [token for token in sent if not is_stopword_punctuation(token)]
    except:
        a=0
    return new_sent

'''
Functions for computing sentence similarity between a set of source sentences and a set of summary sentences

'''
def get_similarity(enc_tokens, summ_tokens, vocab):
    metric = 'precision'
    summ_tokens_combined = flatten_list_of_lists(summ_tokens)
    importances_hat = rouge_l_similarity(enc_tokens, summ_tokens_combined, vocab, metric=metric)
    return importances_hat

# @profile
def rouge_l_similarity(article_sents, abstract_sents, vocab, metric='f1'):
    sentence_similarity = np.zeros([len(article_sents)], dtype=float)
    abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        rouge_l = calc_ROUGE_L_score(article_sent, abstract_sents_removed_periods, metric=metric)
        sentence_similarity[article_sent_idx] = rouge_l
    return sentence_similarity

def rouge_l_similarity_matrix(article_sents, abstract_sents, metric='f1'):
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge_l = calc_ROUGE_L_score(article_sent, abstract_sent, metric=metric)
            abs_similarities.append(rouge_l)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge_l
    return sentence_similarity_matrix

def rouge_1_similarity_matrix(article_sents, abstract_sents, metric, should_remove_stop_words):
    if should_remove_stop_words:
        article_sents = [remove_stopwords_punctuation(sent) for sent in article_sents]
        abstract_sents = [remove_stopwords_punctuation(sent) for sent in abstract_sents]
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge = rouge_functions.rouge_1(article_sent, abstract_sent, 0.5, metric=metric)
            abs_similarities.append(rouge)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge
    return sentence_similarity_matrix

def rouge_2_similarity_matrix(article_sents, abstract_sents, metric, should_remove_stop_words):
    if should_remove_stop_words:
        article_sents = [remove_stopwords_punctuation(sent) for sent in article_sents]
        abstract_sents = [remove_stopwords_punctuation(sent) for sent in abstract_sents]
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge = rouge_functions.rouge_2(article_sent, abstract_sent, 0.5, metric=metric)
            abs_similarities.append(rouge)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge
    return sentence_similarity_matrix


def write_to_temp_files(string_list, temp_dir):
    file_paths = []
    for s_idx, s in enumerate(string_list):
        file_path = os.path.join(temp_dir, '%06d.txt' % s_idx)
        file_paths.append(file_path)
        with open(file_path, 'wb') as f:
            f.write(s)
    return file_paths


def get_doc_substituted_tfidf_matrix(tfidf_vectorizer, sentences, article_text, pca=None):
    # file_paths = write_to_temp_files([article_text], temp_dir)
    # doc_vec = tfidf_vectorizer.transform(file_paths)
    # file_paths = write_to_temp_files(sentences, temp_dir)
    # sent_term_matrix = tfidf_vectorizer.transform(file_paths)

    sent_term_matrix = tfidf_transform_then_pca(tfidf_vectorizer, sentences, pca)

    # doc_vec = tfidf_vectorizer.transform([article_text])
    # sent_term_matrix = tfidf_vectorizer.transform(sentences)
    if pca is None:
        doc_vec = tfidf_transform_then_pca(tfidf_vectorizer, [article_text], pca)
        nonzero_rows, nonzero_cols = sent_term_matrix.nonzero()
        nonzero_indices = list(zip(nonzero_rows, nonzero_cols))
        for idx in nonzero_indices:
            val = doc_vec[0, idx[1]]
            sent_term_matrix[idx] = val
    return sent_term_matrix

def chunk_file(set_name, out_full_dir, out_dir, chunk_size=1000):
  in_file = os.path.join(out_full_dir, '%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(out_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(chunk_size):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def decode_text(text):
    # # print (python_version)
    # # if python_version == 3:
    # #     if isinstance(text, str):
    # #         text = text.encode()
    # #         print ("String: " + str(isinstance(text, str)))
    # #         print ("btyes: " + str(isinstance(text, bytes)))
    # #         return text
    # # else:
    try:
        text = text.decode('utf-8')
    except:
        try:
            text = text.decode('latin-1')
        except:
            raise
    return text

def encode_text(text):
    # # print (python_version)
    # # if python_version == 3:
    # #     if isinstance(text, str):
    # #         text = text.encode()
    # #         print ("String: " + str(isinstance(text, str)))
    # #         print ("btyes: " + str(isinstance(text, bytes)))
    # #         return text
    # # else:
    try:
        text = text.encode('utf-8')
    except:
        try:
            text = text.encode('latin-1')
        except:
            raise
    return text

def unpack_tf_example(example, names_to_types):
    def get_string(name):
        return decode_text(example.features.feature[name].bytes_list.value[0])
    def get_string_list(name):
        texts = get_list(name)
        texts = [decode_text(text) for text in texts]
        return texts
    def get_list(name):
        return example.features.feature[name].bytes_list.value
    def get_delimited_list(name):
        text = get_string(name)
        return text.strip().split(' ')
    def get_delimited_list_of_lists(name, is_string_list=False):
        if not is_string_list:
            text = get_string(name)
        else:
            text = name
        # print (text)
        my_list = text.strip()
        my_list = my_list.split(';')
        return [[int(i) for i in (l.strip().split(' ') if l != '' else [])] for l in my_list]
    def get_delimited_list_of_list_of_lists(name):
        text = get_string(name)
        my_list = text.strip().split('|')
        return [get_delimited_list_of_lists(list_of_lists, is_string_list=True) for list_of_lists in my_list]
    def get_delimited_list_of_tuples(name):
        list_of_lists = get_delimited_list_of_lists(name)
        return [tuple(l) for l in list_of_lists]
    def get_json(name):
        text = get_string(name)
        return json.loads(text)
    func = {'string': get_string,
            'list': get_list,
            'string_list': get_string_list,
            'delimited_list': get_delimited_list,
            'delimited_list_of_lists': get_delimited_list_of_lists,
            'delimited_list_of_list_of_lists': get_delimited_list_of_list_of_lists,
            'delimited_list_of_tuples': get_delimited_list_of_tuples,
            'json': get_json}

    res = []
    for name, type in names_to_types:
        if name not in example.features.feature:
            if name == 'doc_indices':
                res.append(None)
                continue
            else:
                # return [None] * len(names_to_types)
                print(example)
                raise Exception('%s is not a feature of TF Example' % name)
        res.append(func[type](name))
    return res

# def get_tfidf_importances(raw_article_sents, tfidf_model_path=None):
def get_tfidf_importances(tfidf_vectorizer, raw_article_sents, pca=None):
    article_text = ' '.join(raw_article_sents)
    sent_reps = get_doc_substituted_tfidf_matrix(tfidf_vectorizer, raw_article_sents, article_text, pca)
    cluster_rep = np.mean(sent_reps, axis=0).reshape(1, -1)
    similarity_matrix = cosine_similarity(sent_reps, cluster_rep)
    return np.squeeze(similarity_matrix, 1)

def singles_to_singles_pairs(distribution):
    possible_pairs = [tuple(x) for x in
                      list(itertools.combinations(list(range(len(distribution))), 2))]  # all pairs
    possible_singles = [tuple([i]) for i in range(len(distribution))]
    all_combinations = possible_pairs + possible_singles
    out_dict = {}
    for single in possible_singles:
        out_dict[single] = distribution[single[0]]
    for pair in possible_pairs:
        average = (distribution[pair[0]] + distribution[pair[1]]) / 2.0
        out_dict[pair] = average
    return out_dict

def combine_sim_and_imp(logan_similarity, logan_importances, lambda_val=0.6):
    mmr = lambda_val*logan_importances - (1-lambda_val)*logan_similarity
    mmr = np.maximum(mmr, 0)
    return mmr

def combine_sim_and_imp_dict(similarities_dict, importances_dict, lambda_val=0.6):
    mmr = {}
    for key in list(importances_dict.keys()):
        try:
            mmr[key] = combine_sim_and_imp(similarities_dict[key], importances_dict[key], lambda_val=lambda_val)
        except:
            a=0
            raise
    return mmr

def calc_MMR(raw_article_sents, article_sent_tokens, summ_tokens, vocab, importances=None):
    if importances is None:
        importances = get_tfidf_importances(raw_article_sents)
    importances = special_squash(importances)
    similarities = rouge_l_similarity(article_sent_tokens, summ_tokens, vocab, metric='precision')
    mmr = special_squash(combine_sim_and_imp(similarities, importances))
    return mmr

# @profile
def calc_MMR_source_indices(article_sent_tokens, summ_tokens, vocab, importances_dict, qid=None):
    if qid is not None:
        importances_dict = importances_dict[qid]
    importances_dict = special_squash_dict(importances_dict)
    similarities = rouge_l_similarity(article_sent_tokens, summ_tokens, vocab, metric='precision')
    similarities_dict = singles_to_singles_pairs(similarities)
    mmr_dict = special_squash_dict(combine_sim_and_imp_dict(similarities_dict, importances_dict))
    return mmr_dict

def calc_MMR_all(raw_article_sents, article_sent_tokens, summ_sent_tokens, vocab):
    all_mmr = []
    summ_tokens_so_far = []
    mmr = calc_MMR(raw_article_sents, article_sent_tokens, summ_tokens_so_far, vocab)
    all_mmr.append(mmr)
    for summ_sent in summ_sent_tokens:
        summ_tokens_so_far.extend(summ_sent)
        mmr = calc_MMR(raw_article_sents, article_sent_tokens, summ_tokens_so_far, vocab)
        all_mmr.append(mmr)
    all_mmr = np.stack(all_mmr)
    return all_mmr

def special_squash(distribution):
    res = distribution - np.min(distribution)
    if np.max(res) == 0:
        print('All elements in distribution are 0, so setting all to 0')
        res.fill(0)
    else:
        res = res / np.max(res)
    return res

def special_squash_dict(distribution_dict):
    distribution = list(distribution_dict.values())
    values = special_squash(distribution)
    keys = list(distribution_dict.keys())
    items = list(zip(keys, values))
    out_dict = {}
    for key, val in items:
        out_dict[key] = val
    return out_dict

def print_execution_time(start_time):
    localtime = time.asctime( time.localtime(time.time()) )
    print(("Finished at: ", localtime))
    time_taken = time.time() - start_time
    if time_taken < 60:
        print(('Execution time: ', time_taken, ' sec'))
    elif time_taken < 3600:
        print(('Execution time: ', time_taken/60., ' min'))
    else:
        print(('Execution time: ', time_taken/3600., ' hr'))

def split_list_by_item(lst, item):
    return [list(y) for x, y in itertools.groupby(lst, lambda z: z == item) if not x]

def show_callers_locals():
    """Print the local variables in the caller's frame."""
    callers_local_vars = list(inspect.currentframe().f_back.f_back.f_back.f_locals.items())
    return callers_local_vars

def varname(my_var):
    callers_locals = show_callers_locals()
    return [var_name for var_name, var_val in callers_locals if var_val is my_var]

def print_vars(*args):
    for v in args:
        print(varname(v), v)

def reorder(l, ordering):
    return [l[i] for i in ordering]

def shuffle(*args):
    if len(args) == 0:
        raise Exception('No lists to shuffle')
    permutation = np.random.permutation(len(args[0]))
    return [reorder(arg, permutation) for arg in args]

def create_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def reshape_like(to_reshape, thing_with_shape):
    res = []
    if len(to_reshape) != len(flatten_list_of_lists(thing_with_shape)):
        print('Len of to_reshape (' + str(len(to_reshape)) + ') does not equal len of thing_with_shape (' + str(len(flatten_list_of_lists(thing_with_shape))) + ')')
        raise Exception('error')
    idx = 0
    for lst in thing_with_shape:
        list_to_add = []
        for _ in lst:

            try:
                list_to_add.append(to_reshape[idx])
            except:
                a=0
                raise
            idx += 1
        res.append(list_to_add)
    return res

def reshape_like_3d(to_reshape, thing_with_shape):
    res = []
    if len(to_reshape) != len(flatten_list_of_lists_3d(thing_with_shape)):
        print('Len of to_reshape (' + str(len(to_reshape)) + ') does not equal len of thing_with_shape (' + str(len(flatten_list_of_lists_3d(thing_with_shape))) + ')')
        raise Exception('error')
    idx = 0
    for lst in thing_with_shape:
        list_to_add = []
        for l in lst:
            l_to_add = []
            for _ in l:

                try:
                    l_to_add.append(to_reshape[idx])
                except:
                    a=0
                    raise
                idx += 1
            list_to_add.append(l_to_add)
        res.append(list_to_add)
    return res

def enforce_sentence_limit(groundtruth_similar_source_indices_list, sentence_limit):
    enforced_groundtruth_ssi_list = [ssi[:sentence_limit] for ssi in groundtruth_similar_source_indices_list]
    return enforced_groundtruth_ssi_list

def hist_as_pdf_str(hist):
    vals, bins = hist
    length = np.sum(vals)
    pdf = vals * 100.0 / length
    return '%.2f\t'*len(pdf) % (tuple(pdf.tolist())) + '\n' + str(bins)

def find_largest_ckpt_folder(my_dir):
    folder_names = os.listdir(my_dir)
    folder_ckpt_nums = []
    for folder_name in folder_names:
        if '-' not in folder_name:
            ckpt_num = -1
        else:
            ckpt_num = int(folder_name.split('-')[-1].split('_')[0])
        folder_ckpt_nums.append(ckpt_num)
    max_idx = np.argmax(folder_ckpt_nums)
    return folder_names[max_idx]

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def tfidf_transform_then_pca(tfidf_vectorizer, texts, pca=None):
    matrix = tfidf_vectorizer.transform(texts)
    if pca is not None:
        matrix = pca.transform(matrix)
    return matrix

# @profile
def get_first_available_sent(enforced_groundtruth_ssi_list, raw_article_sents, replaced_ssi_list):
    flat_ssi_list = flatten_list_of_lists(enforced_groundtruth_ssi_list + replaced_ssi_list)
    if FLAGS.dataset_name == 'xsum':
        available_range = list(range(1, len(raw_article_sents))) + [0]
    else:
        available_range = list(range(len(raw_article_sents)))
    for sent_idx in available_range:
        if sent_idx not in flat_ssi_list:
            return (sent_idx,)
    return ()       # if we reach here, there are no available sents left

def replace_empty_ssis(enforced_groundtruth_ssi_list, raw_article_sents, sys_alp_list=None):
    replaced_ssi_list = []
    replaced_alp_list = []
    for ssi_idx, ssi in enumerate(enforced_groundtruth_ssi_list):
        if len(ssi) == 0:
            first_available_sent = get_first_available_sent(enforced_groundtruth_ssi_list, raw_article_sents, replaced_ssi_list)
            if len(first_available_sent) != 0:
                replaced_ssi_list.append(first_available_sent)
                chosen_sent = first_available_sent[0]
                alp = [list(range(len(raw_article_sents[chosen_sent].split(' '))))]
                replaced_alp_list.append(alp)
            else:
                a=0     # Don't add the summary sentence because all the source sentences are used up
        else:
            replaced_ssi_list.append(ssi)
            replaced_alp_list.append(sys_alp_list[ssi_idx])
    return replaced_ssi_list, replaced_alp_list

def sent_selection_eval(ssi_list, operation_on_gt):
    if FLAGS.dataset_name == 'cnn_dm':
        sys_max_sent_len = 4
    elif FLAGS.dataset_name == 'duc_2004':
        sys_max_sent_len = 5
    elif FLAGS.dataset_name == 'xsum':
        sys_max_sent_len = 1
    sys_pos = 0
    sys_neg = 0
    gt_pos = 0
    gt_neg = 0
    for gt, sys_, ext_len, article_lcs_paths, _, _ in ssi_list:
        gt = operation_on_gt(gt)
        sys_ = sys_[:sys_max_sent_len]
        sys_ = flatten_list_of_lists(sys_)
        # sys_ = sys_[:ext_len]
        for ssi in sys_:
            if ssi in gt:
                sys_pos += 1
            else:
                sys_neg += 1
        for ssi in gt:
            if ssi in sys_:
                gt_pos += 1
            else:
                gt_neg += 1
    prec = float(sys_pos) / (sys_pos + sys_neg)
    rec = float(gt_pos) / (gt_pos + gt_neg)
    if sys_pos + sys_neg == 0 or gt_pos + gt_neg == 0:
        f1 = 0
    else:
        f1 = 2.0 * (prec * rec) / (prec + rec)
    prec *= 100
    rec *= 100
    f1 *= 100
    suffix = '%.2f\t%.2f\t%.2f\t' % (prec, rec, f1)
    print('Lambdamart P/R/F: ')
    print(suffix)
    return suffix

def word_tag_eval(all_gt_word_tags, all_sys_word_tags):
    result = precision_recall_fscore_support(all_gt_word_tags, all_sys_word_tags)
    suffix = '\t'.join(str(score) for score in result) + '\n'
    return suffix

def all_sent_selection_eval(ssi_list):
    chronological_ssi = True
    def flatten(gt):
        if chronological_ssi:
            gt = make_ssi_chronological(gt)
        return flatten_list_of_lists(gt)
    def primary(gt):
        if chronological_ssi:
            try:
                return [min(ssi) for ssi in gt if len(ssi) > 0]
            except:
                print_vars(gt)
                raise
        else:
            return flatten_list_of_lists(enforce_sentence_limit(gt, 1))
    def secondary(gt):
        if chronological_ssi:
            return [max(ssi) for ssi in gt if len(ssi) == 2]
        else:
            return [ssi[1] for ssi in gt if len(ssi) == 2]
    # def single(gt):
    #     return util.flatten_list_of_lists([ssi for ssi in gt if len(ssi) == 1])
    # def pair(gt):
    #     return util.flatten_list_of_lists([ssi for ssi in gt if len(ssi) == 2])
    operations_on_gt = [flatten, primary, secondary]
    suffixes = []
    for op in operations_on_gt:
        suffix = sent_selection_eval(ssi_list, op)
        suffixes.append(suffix)
    combined_suffix = '\n' + ''.join(suffixes)
    print(combined_suffix)
    return combined_suffix

def lemmatize_sent_tokens(article_sent_tokens):
    # article_sent_tokens_lemma = [[t.lemma_ for t in Doc(nlp.vocab, words=[token.decode('utf-8') for token in sent])] for sent in article_sent_tokens]
    # article_sent_tokens_lemma = [[t.lemma_ for t in Doc(nlp.vocab, words=[decode_text(token) for token in sent])] for sent in article_sent_tokens]
    article_sent_tokens_lemma = [[t.lemma_ for t in Doc(nlp.vocab, words=[token for token in sent])] for sent in article_sent_tokens]

    # article_sent_tokens_lemma2 = [[t.lemma_ for t in nlp2(' '.join(sent))] for sent in article_sent_tokens]
    # for a, b in zip(flatten_list_of_lists(article_sent_tokens), flatten_list_of_lists(article_sent_tokens_lemma)):
    #     if a != b:
    #         print a + '\t' + b

    # for a, b in zip(article_sent_tokens_lemma, article_sent_tokens_lemma2):
    #     if len(a) != len(b):
    #         for j in range(max(len(a) ,len(b))):
    #             if j >= len(a):
    #                 print '\t\t' + b[j]
    #             elif j >= len(b):
    #                 print a[j] + '\t\t'
    #             else:
    #                 print a[j] + '\t' + b[j]
    #         print '\n\n'

    return article_sent_tokens_lemma

average_sents_for_dataset = {
    'cnn_dm': 4,
    'xsum': 1,
    'duc_2004': 5
}

def fix_bracket_token(token):
    if token == '(':
        return '-lrb-'
    elif token == ')':
        return '-rrb-'
    elif token == '[':
        return '-lsb-'
    elif token == ']':
        return '-rsb-'
    else:
        return token

def unfix_bracket_tokens_in_sent(sent):
    return sent.replace('-lrb-', '(').replace('-rrb-', ')').replace('-lsb-', '[').replace('-rsb-', ']').replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']')

def is_quote(tokens):
    contains_quotation_marks = "''" in tokens and len(tokens) > 0 and tokens[0] == "``"
    doesnt_end_with_period = len(tokens) > 0 and tokens[-1] != "."
    # contains_says = "says" in tokens or "said" in tokens
    decision = contains_quotation_marks or doesnt_end_with_period
    # if decision:
    #     print "Skipping quote: ", ' '.join(tokens)
    return decision

def fix_punctuations(text):
    return text.replace(" ''", '"').replace('`` ', '"').replace('` ', "'").replace(" '", "'").replace(' :', ':').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(" 's", "'s").replace(' %', '%').replace('$ ', '$').replace(" 'll", "'ll").replace(" 're", "'re").replace(" n't", "n't")

def process_sent(sent, whitespace=False):
    # line = decode_text(sent.lower())
    line = sent.lower()
    if whitespace:
        tokenized_sent = line.split(' ')
    else:
        tokenized_sent = nltk.word_tokenize(line)
    tokenized_sent = [fix_bracket_token(token) for token in tokenized_sent]
    return tokenized_sent

def make_ssi_chronological(ssi, article_lcs_paths_list=None):
    is_2d = type(ssi[0]) == list or type(ssi[0]) == tuple
    if is_2d:
        new_ssi = []
        new_article_lcs_paths_list = []
        for source_indices_idx, source_indices in enumerate(ssi):
            if article_lcs_paths_list:
                article_lcs_paths = article_lcs_paths_list[source_indices_idx]
            if len(source_indices) >= 2:
                if source_indices[0] > source_indices[1]:
                    source_indices = (min(source_indices), max(source_indices))
                    if article_lcs_paths_list:
                        article_lcs_paths = (article_lcs_paths[1], article_lcs_paths[0])
            new_ssi.append(source_indices)
            if article_lcs_paths_list:
                new_article_lcs_paths_list.append(article_lcs_paths)
        if article_lcs_paths_list:
            return new_ssi, new_article_lcs_paths_list
        else:
            return new_ssi
    else:
        source_indices = ssi
        if article_lcs_paths_list:
            article_lcs_paths = article_lcs_paths_list
        if len(source_indices) >= 2:
            if source_indices[0] > source_indices[1]:
                source_indices = (min(source_indices), max(source_indices))
                if article_lcs_paths_list:
                    article_lcs_paths = (article_lcs_paths[1], article_lcs_paths[0])
        if article_lcs_paths_list:
            return source_indices, article_lcs_paths
        else:
            return source_indices

def filter_pairs_by_sent_position(possible_pairs, rel_sent_indices=None):
    max_sent_position = {
        'cnn_dm': 30,
        'xsum': 20,
        'duc_2004': np.inf
    }
    if FLAGS.dataset_name == 'duc_2004':
        return [pair for pair in possible_pairs if max(rel_sent_indices[pair[0]], rel_sent_indices[pair[1]]) < 5]
    else:
        return [pair for pair in possible_pairs if max(pair) < max_sent_position[FLAGS.dataset_name]]

def get_rel_sent_indices(doc_indices, article_sent_tokens):
    if FLAGS.dataset_name != 'duc_2004' and len(doc_indices) != len(flatten_list_of_lists(article_sent_tokens)):
        doc_indices = [0] * len(flatten_list_of_lists(article_sent_tokens))
    doc_indices_sent_tokens = reshape_like(doc_indices, article_sent_tokens)
    sent_doc = [sent[0] for sent in doc_indices_sent_tokens]
    rel_sent_indices = []
    doc_sent_indices = []
    cur_doc_idx = 0
    rel_sent_idx = 0
    for doc_idx in sent_doc:
        if doc_idx != cur_doc_idx:
            rel_sent_idx = 0
            cur_doc_idx = doc_idx
        rel_sent_indices.append(rel_sent_idx)
        doc_sent_indices.append(cur_doc_idx)
        rel_sent_idx += 1
    doc_sent_lens = [sum(1 for my_doc_idx in doc_sent_indices if my_doc_idx == doc_idx) for doc_idx in range(max(doc_sent_indices) + 1)]
    return rel_sent_indices, doc_sent_indices, doc_sent_lens


def get_indices_of_first_k_sents_of_each_article(rel_sent_indices, k):
    indices = [idx for idx, rel_sent_idx in enumerate(rel_sent_indices) if rel_sent_idx < k]
    return indices

def alp_list_to_binary_tags(alp_list, article_sent_tokens_full, ssi_list):
    tags = []
    for summ_sent_idx, alp in enumerate(alp_list):
        source_indices = ssi_list[summ_sent_idx]
        for priority_idx, source_idx in enumerate(source_indices):
            my_tags = [0] * len(article_sent_tokens_full[source_idx])
            for word_idx in alp_list[summ_sent_idx][priority_idx]:
                my_tags[word_idx] = 1
            tags.extend(my_tags)
    return tags

def get_sent_start_indices(article_sent_tokens):
    indices = []
    cur_idx = 0
    for sent in article_sent_tokens:
        indices.append(cur_idx)
        cur_idx += len(sent)
    return indices

def get_coref_chains(text_a, text_b):
    raw_article_sents = [text_a, text_b]
    article_sent_tokens = [process_sent(sent, whitespace=False) for sent in raw_article_sents]
    all_sent_tokens = article_sent_tokens
    text = ' '.join([' '.join(sent) for sent in all_sent_tokens])
    tokens = text.split(' ')
    sent_start_indices = get_sent_start_indices(all_sent_tokens)
    user_data = {'sent_start_indices': sent_start_indices}
    doc = spacy.tokens.doc.Doc(
        nlp.vocab, tokens, user_data=user_data)
    # run the standard pipeline against it
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    cluster_list = []
    for cluster in doc._.coref_clusters:
        mention_list = []
        contains_mention_in_sent_1 = False
        contains_mention_in_sent_2 = False
        for mention in cluster.mentions:
            sent_idx, start_idx, end_idx = get_sent_and_word_idx_of_mention(mention, all_sent_tokens)
            if sent_idx == 1:
                start_idx += len(all_sent_tokens[0])
                end_idx += len(all_sent_tokens[0])
                contains_mention_in_sent_2 = True
            else:
                contains_mention_in_sent_1 = True

            mention_list.append([start_idx, end_idx])
        if contains_mention_in_sent_1 and contains_mention_in_sent_2:
            cluster_list.append(mention_list)
    # new_corefs_str = json.dumps(cluster_list)
    print(cluster_list)

    return cluster_list, article_sent_tokens

def make_tf_example(simple_similar_source_indices, raw_article_sents, summary_text, corefs_str, doc_indices,
                    article_lcs_paths_list, coref_chains, coref_representatives):
    tf_example = example_pb2.Example()
    source_indices_str = ';'.join([' '.join(str(i) for i in source_indices) for source_indices in simple_similar_source_indices])
    tf_example.features.feature['similar_source_indices'].bytes_list.value.extend([encode_text(source_indices_str)])
    for sent in raw_article_sents:
        s = sent.strip()
        tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([encode_text(s)])
    if FLAGS.dataset_name == 'duc_2004':
        for summ_text in summary_text:
            tf_example.features.feature['summary_text'].bytes_list.value.extend([encode_text(summ_text)])
    else:
        tf_example.features.feature['summary_text'].bytes_list.value.extend([encode_text(summary_text)])
    if doc_indices is not None:
        tf_example.features.feature['doc_indices'].bytes_list.value.extend([encode_text(doc_indices)])
    if corefs_str is not None:
        tf_example.features.feature['corefs'].bytes_list.value.extend([encode_text(corefs_str)])
    if article_lcs_paths_list is not None:
        article_lcs_paths_list_str = '|'.join([';'.join([' '.join(str(i) for i in source_indices) for source_indices in article_lcs_paths]) for article_lcs_paths in article_lcs_paths_list])
        tf_example.features.feature['article_lcs_paths_list'].bytes_list.value.extend([encode_text(article_lcs_paths_list_str)])
    if coref_chains is not None:
        coref_chains_str = '|'.join([';'.join([' '.join(str(i) for i in mention) for mention in chain]) for chain in coref_chains])
        tf_example.features.feature['coref_chains'].bytes_list.value.extend([encode_text(coref_chains_str)])
    if coref_representatives is not None:
        for rep in coref_representatives:
            tf_example.features.feature['coref_representatives'].bytes_list.value.extend([encode_text(rep)])
    return tf_example

def write_tf_example(writer, tf_example):
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def coref_replace(raw_article_sents, ssi, coref_chains, coref_representatives):
    raw_article_sent_tokens = [sent.split(' ') for sent in reorder(raw_article_sents, ssi)]
    already_touched = [ [False]*len(sent) for sent in raw_article_sent_tokens ]
    coref_chains_flat = []
    for chain_idx, chain in enumerate(coref_chains):
        rep_words = coref_representatives[chain_idx].split(' ')
        for loc in chain:
            new_loc = tuple(list(loc) + [rep_words])
            coref_chains_flat.append(new_loc)
    coref_chains_flat.sort(key=lambda x: x[2])
    coref_chains_flat = coref_chains_flat[::-1]
    for loc in coref_chains_flat:
        sent_idx, start, end, rep_words = loc
        span_already_touched = False
        for word_idx in range(start, end):
            if already_touched[sent_idx][word_idx]:
                span_already_touched = True
                break
        if not span_already_touched:
            raw_article_sent_tokens[sent_idx][start:end] = rep_words
            for word_idx in range(start, end):
                already_touched[sent_idx][word_idx] = True
    return raw_article_sent_tokens

def rouge_significantly_better(my_r1, my_r2, my_rl, o_r1, o_r2, o_rl):
    return my_r1 - o_r1 >= 0.02 and my_r2 - o_r2 >= 0.02 and my_rl - o_rl >= 0.02

def rouge_significantly_worse(my_r1, my_r2, my_rl, o_r1, o_r2, o_rl):
    return my_r1 - o_r1 <= -0.02 and my_r2 - o_r2 <= -0.02 and my_rl - o_rl <= -0.02

def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)



















