import copy
from tqdm import tqdm
from scoop import futures
import rouge_functions
from absl import flags
from absl import app
import convert_data
import time
import subprocess
import itertools
import glob
import numpy as np
import data
import os
import sys
from collections import defaultdict
import util
from scipy import sparse
from ssi_functions import html_highlight_sents_in_article, get_simple_source_indices_list
import pickle
import ssi_functions
# from profilestats import profile

if 'dataset_name' in flags.FLAGS:
    flags_already_done = True
else:
    flags_already_done = False

FLAGS = flags.FLAGS
if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'start_over' not in flags.FLAGS:
    flags.DEFINE_boolean('start_over', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'first_k' not in flags.FLAGS:
    flags.DEFINE_integer('first_k', 20, 'Specifies k, where we consider only the first k sentences of each article. Only applied when [running on both singles and pairs, and not running on cnn_dm]')
if 'upper_bound' not in flags.FLAGS:
    flags.DEFINE_boolean('upper_bound', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'use_pair_criteria' not in flags.FLAGS:
    flags.DEFINE_boolean('use_pair_criteria', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'pca' not in flags.FLAGS:
    flags.DEFINE_boolean('pca', False, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'sent_position_criteria' not in flags.FLAGS:
    flags.DEFINE_boolean('sent_position_criteria', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'sentemb' not in flags.FLAGS:
    flags.DEFINE_boolean('sentemb', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'artemb' not in flags.FLAGS:
    flags.DEFINE_boolean('artemb', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
if 'plushidden' not in flags.FLAGS:
    flags.DEFINE_boolean('plushidden', True, 'Which mode to run in. Must be in {write_to_file, generate_summaries}.')
# flags.DEFINE_boolean('l_sents', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_bool("tag_tokens", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_float("tag_loss_wt", 0.2, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("use_mmr", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_float("tag_threshold", 0.2, "What threshold to use for choosing tags. Only applicable if binarize_method == 'threshold'")
flags.DEFINE_integer("summ_limit", 100, "What limit to use for choosing tags. Only applicable if binarize_method == 'summ_limit'")
flags.DEFINE_integer("inst_limit", 10, "What limit to use for choosing tags. Only applicable if binarize_method == 'inst_limit'")
flags.DEFINE_string("binarize_method", "threshold", "Which method to use for turning token tag probabilities into binary tags. Can be one of {threshold, summ_limit, inst_limit}.")
flags.DEFINE_bool('use_val_test', False, 'Which dataset split to use. Must be one of {train, val, test}')

if not flags_already_done:
    FLAGS(sys.argv)

_exp_name = 'bert'
if FLAGS.pca:
    model += '_pca'
tfidf_model = 'all'
importance = True
filter_sentences = True
num_instances = -1
random_seed = 123
max_sent_len_feat = 20
min_matched_tokens = 2
# singles_and_pairs = 'singles'
include_tfidf_vec = True
if FLAGS.use_val_test:
    dataset_split = 'val_test'
else:
    dataset_split = 'test'

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens'
bert_in_dir = os.path.join('data', 'bert', FLAGS.dataset_name, FLAGS.singles_and_pairs, 'input')
bert_scores_dir = os.path.join('data', 'bert', FLAGS.dataset_name, FLAGS.singles_and_pairs, 'output')
ssi_out_dir = 'data/temp/' + FLAGS.dataset_name + '/ssi'
log_dir = 'logs'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]

if FLAGS.singles_and_pairs == 'both':
    exp_name = FLAGS.dataset_name + '_' + _exp_name + '_both'
    dataset_articles = FLAGS.dataset_name
else:
    exp_name = FLAGS.dataset_name + '_' + _exp_name + '_singles'
    dataset_articles = FLAGS.dataset_name + '_singles'

# if FLAGS.sentemb:
#     exp_name += '_sentemb'
#     bert_scores_dir += '_sentemb'
# if FLAGS.artemb:
#     exp_name += '_artemb'
#     bert_scores_dir += '_artemb'
# if FLAGS.plushidden:
#     exp_name += '_plushidden'
#     bert_scores_dir += '_plushidden'
if FLAGS.tag_tokens:
    exp_name += '_tag' + str(FLAGS.tag_loss_wt)
    bert_scores_dir += '_tag' + str(FLAGS.tag_loss_wt)

if FLAGS.use_mmr:
    exp_name += '_mmr'

if FLAGS.upper_bound:
    exp_name = exp_name + '_upperbound'

if FLAGS.pca:
    exp_name = exp_name + '_pca'


if FLAGS.singles_and_pairs == 'singles':
    sentence_limit = 1
else:
    sentence_limit = 2

# if FLAGS.dataset_name == 'xsum':
#     l_param = 40
# else:
#     l_param = 100
l_param = 100

if FLAGS.pca:
    bert_in_dir += '_pca'
    bert_scores_dir += '_pca'
temp_in_path = os.path.join(bert_in_dir, dataset_split + '.tsv')
temp_out_path = os.path.join(bert_scores_dir, dataset_split + '_results.tsv')
file_path_seq = os.path.join(bert_scores_dir, dataset_split + '_results_seq.tsv')
file_path_mappings = os.path.join(bert_scores_dir, dataset_split + '_results_mappings.tsv')
util.create_dirs(bert_scores_dir)
my_log_dir = os.path.join(log_dir, exp_name)
dec_dir = os.path.join(my_log_dir, 'decoded')
ref_dir = os.path.join(my_log_dir, 'reference')
html_dir = os.path.join(my_log_dir, 'hightlighted_html')
util.create_dirs(dec_dir)
util.create_dirs(ref_dir)
util.create_dirs(html_dir)
util.create_dirs(ssi_out_dir)

# @profile
def read_bert_scores(file_path):
    print ('Reading read_bert_scores')
    with open(file_path) as f:
        lines = f.readlines()
    data = [[float(x) for x in line.split('\t')] for line in tqdm(lines)]
    data = np.array(data)
    return data

# @profile
def read_bert_scores_seq(file_path_seq, file_path_mappings):
    print ('Reading read_bert_scores_seq')
    with open(file_path_seq) as f:
        lines = f.readlines()
    # lines = lines[:10000]
    data = [[[float(x) for x in section.split('\t')] for section in line.split('\t\t')] for line in tqdm(lines)]
    data = np.array(data)
    with open(file_path_mappings) as f:
        lines = f.readlines()
    mappings_data = [[int(x) for x in line.split('\t')] for line in lines]
    return data, mappings_data

# @profile
def get_qid_source_indices(line):
    items = line.split('\t')
    qid = int(items[3])
    inst_id = int(items[4])
    source_indices = [int(x) for x in items[5].split()]

    if len(source_indices) == 2:
        source_indices = [min(source_indices), max(source_indices)]

    return qid, inst_id, source_indices

# @profile
def read_source_indices_from_bert_input(file_path):
    print ('Reading source indices from bert input')
    out_list = []
    with open(file_path) as f:
        lines = f.readlines()[1:]
    for line in tqdm(lines):
        qid, inst_id, source_indices = get_qid_source_indices(line)
        out_list.append(tuple((qid, tuple(source_indices))))
    return out_list

# @profile
def get_sent_or_sents(article_sent_tokens, source_indices):
    chosen_sent_tokens = [article_sent_tokens[idx] for idx in source_indices]
    # sents = util.flatten_list_of_lists(chosen_sent_tokens)
    return chosen_sent_tokens

# @profile
def get_bert_scores_for_singles_pairs(data, qid_source_indices_list):
    print ('get_bert_scores_for_singles_pairs')
    out_dict = {}
    for row_idx, row in enumerate(tqdm(data)):
        score0, score1 = row
        qid, source_indices = qid_source_indices_list[row_idx]
        if qid not in out_dict:
            out_dict[qid] = {}
        out_dict[qid][source_indices] = score1
    return out_dict

def get_token_scores_and_mappings(data, data_mappings, qid_source_indices_list):
    out_dict = {}
    for row_idx, row in enumerate(data):
        tokens_score_list = row
        tokens_mapping_list = data_mappings[row_idx]
        if len(tokens_score_list) != len(tokens_mapping_list):
            raise Exception('Len of tokens_score_list %d != Len of tokens_mapping_list %d' % (len(tokens_score_list), len(tokens_mapping_list)))
        token_scores = [score1 for score0,score1 in tokens_score_list]
        qid, source_indices = qid_source_indices_list[row_idx]
        if qid not in out_dict:
            out_dict[qid] = {}
        out_dict[qid][source_indices] = (token_scores, tokens_mapping_list)
    return out_dict

# @profile
def rank_source_sents(temp_in_path, temp_out_path):
    qid_source_indices_list = read_source_indices_from_bert_input(temp_in_path)
    data = read_bert_scores(temp_out_path)
    if len(qid_source_indices_list) != len(data):
        raise Exception('Len of qid_source_indices_list %d != Len of data %d' % (len(qid_source_indices_list), len(data)))
    source_indices_to_scores = get_bert_scores_for_singles_pairs(data, qid_source_indices_list)
    return source_indices_to_scores

def get_token_scores_for_ssi(temp_in_path, file_path_seq, file_path_mappings):
    qid_source_indices_list = read_source_indices_from_bert_input(temp_in_path)
    data, data_mappings = read_bert_scores_seq(file_path_seq, file_path_mappings)
    if len(qid_source_indices_list) != len(data):
        raise Exception('Len of qid_source_indices_list %d != Len of data %d' % (len(qid_source_indices_list), len(data)))
    if len(qid_source_indices_list) != len(data_mappings):
        raise Exception('Len of qid_source_indices_list %d != Len of data_mappings %d' % (len(qid_source_indices_list), len(data_mappings)))
    source_indices_to_token_scores_and_mappings = get_token_scores_and_mappings(data, data_mappings, qid_source_indices_list)
    return source_indices_to_token_scores_and_mappings

# @profile
def get_best_source_sents(article_sent_tokens, mmr_dict, already_used_source_indices):
    if len(already_used_source_indices) == 0:
        source_indices = max(mmr_dict, key=mmr_dict.get)
    else:
        best_value = -9999999
        best_source_indices = ()
        for key, val in mmr_dict.items():
            if val > best_value and not any(i in list(key) for i in already_used_source_indices):
                best_value = val
                best_source_indices = key
        source_indices = best_source_indices
    sents = get_sent_or_sents(article_sent_tokens, source_indices)
    return sents, source_indices

def get_token_info_for_ssi(qid_ssi_to_token_scores_and_mappings, qid, source_indices):
    return qid_ssi_to_token_scores_and_mappings[qid][source_indices]

def consolidate_token_scores(token_scores, token_mappings):
    token_cons_scores = []
    cur_sent_token_scores = []
    prev_mapping = -3
    for token_idx, score in enumerate(token_scores):
        mapping = token_mappings[token_idx]
        if mapping == -2:   # token is padding
            prev_mapping = mapping
            continue
        elif mapping == -1:
            if token_idx == 0:   # token is [CLS]
                prev_mapping = mapping
                continue
            else:   # token is [SEP], so it means we finished a sentence
                token_cons_scores.append(cur_sent_token_scores)
                cur_sent_token_scores = []
                prev_mapping = mapping
        else:   # token is a real WordPiece token
            if prev_mapping == mapping:     # this token is part of the previous full token
                cur_sent_token_scores[-1] = max(cur_sent_token_scores[-1], score)
            else:   # this token is a new full token
                cur_sent_token_scores.append(score)
                prev_mapping = mapping

    if len(cur_sent_token_scores) != 0:
        print (token_scores, token_mappings)
        raise Exception('Didnt flush out sentence (see printed above)')
    return token_cons_scores


def threshold_token_scores(token_cons_scores, threshold):
    token_tags = [[1 if score >= threshold else 0 for score in sent] for sent in token_cons_scores]
    return token_tags

def filter_untagged(sents, token_tags):
    sents_only_tagged = []
    for sent_idx, sent in enumerate(sents):
        cur_token_tags = token_tags[sent_idx]
        new_sent = [token for token_idx, token in enumerate(sent) if cur_token_tags[token_idx]]
        sents_only_tagged.append(new_sent)
    return sents_only_tagged

def consolidate_and_pad_token_scores(token_scores, token_mappings, sents):
    token_cons_scores = consolidate_token_scores(token_scores, token_mappings)
    if len(token_cons_scores) != len(sents):
        print (token_cons_scores, sents)
        raise Exception('Len of token_cons_scores %d != Len of sents %d' % (len(token_cons_scores), len(sents)))
    padded_token_cons_scores = []       # we need to pad it, because sometimes the instance was too long for BERT, so it got truncated. So we need to fill the end of the sentences with 0 probabilities.
    for sent_idx, sent_scores in enumerate(token_cons_scores):
        sent = sents[sent_idx]
        if len(sent_scores) > len(sent):
            print (token_cons_scores, sents)
            raise Exception('Len of sent_scores %d > Len of sent %d' % (len(sent_scores), len(sent)))
        while len(sent_scores) < len(sent):
            sent_scores.append(0.)
        padded_token_cons_scores.append(sent_scores)
    return padded_token_cons_scores

def generate_summary(article_sent_tokens, qid_ssi_to_importances, example_idx, qid_ssi_to_token_scores_and_mappings):
    qid = example_idx

    summary_sent_tokens = []
    summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
    already_used_source_indices = []
    similar_source_indices_list = []
    summary_sents_for_html = []
    article_lcs_paths_list = []
    token_probs_list = []
    ssi_length_extractive = None
    while len(summary_tokens) < 300:
        if len(summary_tokens) >= l_param and ssi_length_extractive is None:
            ssi_length_extractive = len(similar_source_indices_list)
        # if FLAGS.dataset_name == 'xsum' and len(summary_tokens) > 0:
        #     ssi_length_extractive = len(similar_source_indices_list)
        #     break
        if FLAGS.use_mmr:
            score_dict = util.calc_MMR_source_indices(article_sent_tokens, summary_tokens, None, qid_ssi_to_importances, qid=qid)
        else:
            score_dict = qid_ssi_to_importances[qid]
        sents, source_indices = get_best_source_sents(article_sent_tokens, score_dict, already_used_source_indices)
        if len(source_indices) == 0:
            break

        if qid == 47 and 3 in source_indices and 5 in source_indices:
            a=0
        try:
            token_scores, token_mappings = get_token_info_for_ssi(qid_ssi_to_token_scores_and_mappings, qid, source_indices)
        except:
            print(qid, source_indices, qid_ssi_to_token_scores_and_mappings[qid])
            raise
        # if np.max(token_mappings) !=
        padded_token_cons_scores = consolidate_and_pad_token_scores(token_scores, token_mappings, sents)
        token_probs_list.append(padded_token_cons_scores)
        token_tags = threshold_token_scores(padded_token_cons_scores, FLAGS.tag_threshold)     # shape (1 or 2, len(sent)) 1 or 2 depending on if it is singleton/pair
        article_lcs_paths = ssi_functions.binary_tags_to_list(token_tags)
        article_lcs_paths_list.append(article_lcs_paths)

        # if FLAGS.tag_tokens and FLAGS.tag_loss_wt != 0:
        #     sents_only_tagged = filter_untagged(sents, token_tags)
        #     summary_sent_tokens.extend(sents_only_tagged)
        # else:
        summary_sent_tokens.extend(sents)

        summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
        similar_source_indices_list.append(source_indices)
        summary_sents_for_html.append(' <br> '.join([' '.join(sent) for sent in sents]))
        if filter_sentences:
            already_used_source_indices.extend(source_indices)
    if ssi_length_extractive is None:
        ssi_length_extractive = len(similar_source_indices_list)
    selected_article_sent_indices = util.flatten_list_of_lists(similar_source_indices_list[:ssi_length_extractive])
    summary_sents = [' '.join(sent) for sent in util.reorder(article_sent_tokens, selected_article_sent_indices)]
    # summary = '\n'.join([' '.join(tokens) for tokens in summary_sent_tokens])
    return summary_sents, similar_source_indices_list, summary_sents_for_html, ssi_length_extractive, article_lcs_paths_list, token_probs_list

def example_generator_extended(example_generator, total, qid_ssi_to_importances, qid_ssi_to_token_scores_and_mappings):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
    # for example in example_generator:
        example_idx += 1
        if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
            break
        yield (example, example_idx, qid_ssi_to_importances, qid_ssi_to_token_scores_and_mappings)

# @profile
def write_highlighted_html(html, out_dir, example_idx):
    html = '''
    
<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d_highlighted.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d_highlighted.html";
    };
    
    document.addEventListener("keyup",function(e){
   var key = e.which||e.keyCode;
   switch(key){
      //left arrow
      case 37:
         document.getElementById("btnPrev").click();
      break;
      //right arrow
      case 39:
         document.getElementById("btnNext").click();
      break;
   }
});
</script>

''' % (example_idx-1, example_idx+1) + html
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    with open(path, 'w') as f:
        f.write(html)

def get_indices_of_first_k_sents_of_each_article(rel_sent_indices, k):
    indices = [idx for idx, rel_sent_idx in enumerate(rel_sent_indices) if rel_sent_idx < k]
    return indices

def evaluate_example(ex):
    example, example_idx, qid_ssi_to_importances, qid_ssi_to_token_scores_and_mappings = ex
    print(example_idx)
    # example_idx += 1
    qid = example_idx
    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices, groundtruth_article_lcs_paths_list = util.unpack_tf_example(example, names_to_types)
    article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
    groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
    groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]
    groundtruth_similar_source_indices_list, groundtruth_article_lcs_paths_list = util.make_ssi_chronological(groundtruth_similar_source_indices_list, groundtruth_article_lcs_paths_list)

    if example_idx == 239:
        a = 0
    gt_word_tags = []
    all_padded_token_cons_scores = []
    for summ_sent_idx, source_indices in enumerate(groundtruth_similar_source_indices_list):
        if len(source_indices) == 0:
            all_padded_token_cons_scores.append([])
            continue
        for priority_idx, source_idx in enumerate(source_indices):
            my_gt_tags = [0] * len(article_sent_tokens[source_idx])
            for word_idx in groundtruth_article_lcs_paths_list[summ_sent_idx][priority_idx]:
                my_gt_tags[word_idx] = 1
            gt_word_tags.extend(my_gt_tags)

        sents = util.reorder(article_sent_tokens, source_indices)
        token_scores, token_mappings = get_token_info_for_ssi(qid_ssi_to_token_scores_and_mappings, qid, source_indices)
        padded_token_cons_scores = consolidate_and_pad_token_scores(token_scores, token_mappings, sents)
        all_padded_token_cons_scores.append(padded_token_cons_scores)

    sys_word_tags = []
    binarize_parameter = FLAGS.tag_threshold if FLAGS.binarize_method == 'threshold' else FLAGS.summ_limit if FLAGS.binarize_method == 'summ_limit' else FLAGS.inst_limit if FLAGS.binarize_method == 'inst_limit' else None
    sys_alp_list = ssi_functions.list_labels_from_probs(all_padded_token_cons_scores, FLAGS.binarize_method, binarize_parameter)
    if example_idx < 2:
        print (sys_alp_list)
    for summ_sent_idx, source_indices in enumerate(groundtruth_similar_source_indices_list):
        if len(source_indices) == 0:
            continue
        for priority_idx, source_idx in enumerate(source_indices):
            my_sys_tags = [0] * len(article_sent_tokens[source_idx])
            if summ_sent_idx >= len(sys_alp_list):
                print ("sys_alp_list too small")
                print(summ_sent_idx, priority_idx, sys_alp_list, groundtruth_similar_source_indices_list, article_sent_tokens)
            if priority_idx >= len(sys_alp_list[summ_sent_idx]):
                print ("sys_alp_list[] too small")
                print(summ_sent_idx, priority_idx, sys_alp_list, groundtruth_similar_source_indices_list, article_sent_tokens)
            for word_idx in sys_alp_list[summ_sent_idx][priority_idx]:
                my_sys_tags[word_idx] = 1
            sys_word_tags.extend(my_sys_tags)


    if FLAGS.upper_bound:
        replaced_ssi_list = util.replace_empty_ssis(groundtruth_similar_source_indices_list, raw_article_sents)
        selected_article_sent_indices = util.flatten_list_of_lists(replaced_ssi_list)
        summary_sents = [' '.join(sent) for sent in util.reorder(article_sent_tokens, selected_article_sent_indices)]
        similar_source_indices_list = groundtruth_similar_source_indices_list
        ssi_length_extractive = len(similar_source_indices_list)
    else:
        summary_sents, similar_source_indices_list, summary_sents_for_html, ssi_length_extractive, \
            article_lcs_paths_list, token_probs_list = generate_summary(article_sent_tokens, qid_ssi_to_importances, example_idx, qid_ssi_to_token_scores_and_mappings)
        similar_source_indices_list_trunc = similar_source_indices_list[:ssi_length_extractive]
        summary_sents_for_html_trunc = summary_sents_for_html[:ssi_length_extractive]
        if example_idx < 100 or (example_idx >= 2000 and example_idx < 2100):
            summary_sent_tokens = [sent.split(' ') for sent in summary_sents_for_html_trunc]
            if FLAGS.tag_tokens and FLAGS.tag_loss_wt != 0:
                lcs_paths_list_param = copy.deepcopy(article_lcs_paths_list)
            else:
                lcs_paths_list_param = None
            extracted_sents_in_article_html = html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list_trunc,
                                            article_sent_tokens, doc_indices=doc_indices, lcs_paths_list=lcs_paths_list_param)
            # write_highlighted_html(extracted_sents_in_article_html, html_dir, example_idx)

            groundtruth_ssi_list, gt_lcs_paths_list, gt_article_lcs_paths_list, gt_smooth_article_paths_list = get_simple_source_indices_list(
                                            groundtruth_summ_sent_tokens,
                                           article_sent_tokens, None, sentence_limit, min_matched_tokens)
            groundtruth_highlighted_html = html_highlight_sents_in_article(groundtruth_summ_sent_tokens, groundtruth_ssi_list,
                                            article_sent_tokens, lcs_paths_list=gt_lcs_paths_list, article_lcs_paths_list=gt_smooth_article_paths_list, doc_indices=doc_indices)

            all_html = '<u>System Summary</u><br><br>' + extracted_sents_in_article_html + '<u>Groundtruth Summary</u><br><br>' + groundtruth_highlighted_html
            # all_html = '<u>System Summary</u><br><br>' + extracted_sents_in_article_html
            write_highlighted_html(all_html, html_dir, example_idx)
    rouge_functions.write_for_rouge(groundtruth_summ_sents, summary_sents, example_idx, ref_dir, dec_dir)
    return (groundtruth_similar_source_indices_list, similar_source_indices_list, ssi_length_extractive, token_probs_list, gt_word_tags, all_padded_token_cons_scores, sys_word_tags)


def main(unused_argv):
# def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    print('Running statistics on %s' % exp_name)

    start_time = time.time()
    np.random.seed(random_seed)
    source_dir = os.path.join(data_dir, dataset_articles)
    source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))
    ex_sents = ['single .', 'sentence .']
    article_text = ' '.join(ex_sents)


    total = len(source_files)*1000
    example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False, should_check_valid=False)


    qid_ssi_to_importances = rank_source_sents(temp_in_path, temp_out_path)
    qid_ssi_to_token_scores_and_mappings = get_token_scores_for_ssi(temp_in_path, file_path_seq, file_path_mappings)
    ex_gen = example_generator_extended(example_generator, total, qid_ssi_to_importances, qid_ssi_to_token_scores_and_mappings)
    print('Creating list')
    ex_list = [ex for ex in ex_gen]
    ssi_list = list(futures.map(evaluate_example, ex_list))


    all_gt_word_tags = []
    all_sys_word_tags = []
    for item in ssi_list:
        all_gt_word_tags.extend(item[4])
        all_sys_word_tags.extend(item[6])
    ssi_list = [item[:6] for item in ssi_list]

    # save ssi_list
    with open(os.path.join(my_log_dir, 'ssi.pkl'), 'wb') as f:
        pickle.dump(ssi_list, f)
    with open(os.path.join(my_log_dir, 'ssi.pkl'), 'rb') as f:
        ssi_list = pickle.load(f)
    print('Evaluating BERT model F1 score...')
    suffix = util.word_tag_eval(all_gt_word_tags, all_sys_word_tags)
    suffix += util.all_sent_selection_eval(ssi_list)
    #
    # # for ex in tqdm(ex_list, total=total):
    # #     load_and_evaluate_example(ex)
    #
    print('Evaluating ROUGE...')
    results_dict = rouge_functions.rouge_eval(ref_dir, dec_dir, l_param=l_param)
    # print("Results_dict: ", results_dict)
    rouge_functions.rouge_log(results_dict, my_log_dir, suffix=suffix)

    ssis_restricted = [ssi_triple[1][:ssi_triple[2]] for ssi_triple in ssi_list]
    ssi_lens = [len(source_indices) for source_indices in util.flatten_list_of_lists(ssis_restricted)]
    # print ssi_lens
    num_singles = ssi_lens.count(1)
    num_pairs = ssi_lens.count(2)
    print ('Percent singles/pairs: %.2f %.2f' % (num_singles*100./len(ssi_lens), num_pairs*100./len(ssi_lens)))

    util.print_execution_time(start_time)


if __name__ == '__main__':
    # main()
    app.run(main)




































