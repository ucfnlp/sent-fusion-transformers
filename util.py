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

import time
import os
import numpy as np
from absl import flags
import itertools
from nltk.corpus import stopwords
import string
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English
import sys
import nltk

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

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def flatten_list_of_lists_3d(list_of_list_of_lists):
    return list(itertools.chain.from_iterable([itertools.chain.from_iterable(list_of_lists) for list_of_lists in list_of_list_of_lists]))

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

def to_unicode(text):
    try:
        text = str(text, errors='replace')
    except TypeError:
        return text
    return text

def create_token_to_indices(lst):
    token_to_indices = {}
    for token_idx, token in enumerate(lst):
        if token in token_to_indices:
            token_to_indices[token].append(token_idx)
        else:
            token_to_indices[token] = [token_idx]
    return token_to_indices

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


def decode_text(text):
    try:
        text = text.decode('utf-8')
    except:
        try:
            text = text.decode('latin-1')
        except:
            raise
    return text

def encode_text(text):
    try:
        text = text.encode('utf-8')
    except:
        try:
            text = text.encode('latin-1')
        except:
            raise
    return text

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

def lemmatize_sent_tokens(article_sent_tokens):
    article_sent_tokens_lemma = [[t.lemma_ for t in Doc(nlp.vocab, words=[token for token in sent])] for sent in article_sent_tokens]
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

def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)



















