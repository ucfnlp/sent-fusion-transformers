import itertools
import os
import csv
import numpy as np
import operator
import math
import glob
from sklearn.cluster import DBSCAN
import string
import re
from collections import Counter
from nltk.metrics.agreement import AnnotationTask
from tqdm import tqdm

out_folder = os.path.join('mturk','main_task','processed')
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
out_file = os.path.join(out_folder, 'PoC.tsv')

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

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(i)
    return matches

def find_all(s, subs):
    return [m.start() for m in re.finditer(subs, s)]

def without_punctuation(s):
    # return s.translate(string.maketrans('', ''), string.punctuation)
    new_s = ''
    new_s_indices = []
    removed_s = ''
    removed_s_indices = []

    for ch_idx, ch in enumerate(s):
        if ch == ' ' or ch in string.punctuation:
            removed_s += ch
            removed_s_indices.append(ch_idx)
        else:
            new_s += ch
            new_s_indices.append(ch_idx)
    return new_s, new_s_indices, removed_s, removed_s_indices

def character_to_word_level(tokens, longest_range):
    if longest_range == (0,0):
        return longest_range
    new_range = (-1,-1)
    cur_char = 0
    cur_idx = 0
    token_end_indices = []
    for token in tokens:
        if token == '':
            print(tokens)
            raise Exception('no token should be empty. Probably means there were two consecutive whitespace (" ") characters')
        cur_idx += len(token)
        token_end_indices.append(cur_idx)
        cur_idx += 1
    for word_idx, token_end_idx in enumerate(token_end_indices):
        if longest_range[0] < token_end_idx and new_range[0] == -1:
            new_range = (word_idx, new_range[1])
        if longest_range[1] <= token_end_idx and new_range[1] == -1:
            new_range = (new_range[0], word_idx+1)
        # import pdb; pdb.set_trace()
    if token_end_indices[-1] == longest_range[1] and new_range[1] == -1:
        new_range = (new_range[0], len(tokens))
    # import pdb; pdb.set_trace()
    if new_range[1] <= new_range[0]:
        print(tokens)
        print(longest_range)
        print(new_range)
        # import pdb; pdb.set_trace()
        raise Exception('range end was less than range start')
    return new_range

def fix_punctuation(sloppy_tokens, word_range):
    suffixes = ["'s", "'ll", "'re", "n't"]
    punct_suffixes = [",", ".", "!", "?", '"', "'", ")", "%", ":"]
    punct_prefixes = ["(", '"', "'", "$"]
    resulting_tokens = ["''", "``", "`", "'"]
    all_special_tokens = suffixes + punct_suffixes + punct_prefixes + resulting_tokens
    fixed_tokens = [[token] for token in sloppy_tokens]
    new_range = word_range
    iterations = 4
    for _ in range(iterations):
        for token_idx, token in enumerate(fixed_tokens):
            new_token = []
            for token_piece in token:
                already_suffix = False
                already_prefix = False
                for suffix in suffixes:
                    if not already_suffix and not already_prefix and token_piece.endswith(suffix) and token_piece != suffix and token_piece not in all_special_tokens and token_piece != "''":
                        new_token.append(token_piece[:-len(suffix)])
                        new_token.append(suffix)
                        already_suffix = True
                for suffix in punct_suffixes:
                    if '.' in token_piece and suffix == '.' and 'African' in token_piece:
                        a=0
                    if not already_suffix and not already_prefix and token_piece.endswith(suffix) and token_piece != suffix and token_piece not in all_special_tokens and token_piece != "''":
                        if suffix == "'":
                            a=0
                        if suffix == '"':
                            new_suffix = "''"
                        else:
                            new_suffix = suffix
                        new_token.append(token_piece[:-len(suffix)])
                        new_token.append(new_suffix)
                        already_suffix = True
                for prefix in punct_prefixes:
                    if not already_suffix and not already_prefix and token_piece.startswith(prefix) and token_piece != prefix and token_piece not in all_special_tokens and token_piece != "''":
                        if prefix == '"':
                            new_prefix = "``"
                        elif prefix == "'":
                            new_prefix = "`"
                        else:
                            new_prefix = prefix
                        new_token.append(new_prefix)
                        new_token.append(token_piece[len(prefix):])
                        already_prefix = True
                if not already_prefix and not already_suffix:
                    new_token.append(token_piece)
            fixed_tokens[token_idx] = new_token
    if 'ultraviolet' in sloppy_tokens:
        a=0
    if word_range == (0,0):
        new_range = (0,0)
    else:
        new_range = (-1,-1)
        cur_idx = 0
        for token_idx, token in enumerate(fixed_tokens):
            for in_token_idx, in_token in enumerate(token):
                if word_range[0] == token_idx and new_range[0] == -1 and in_token not in all_special_tokens:
                    new_range = (cur_idx, new_range[1])
                if (word_range[1]-1) == token_idx and in_token not in all_special_tokens:
                    new_range = (new_range[0], cur_idx + 1)
                cur_idx += 1
    new_tokens = flatten_list_of_lists(fixed_tokens)
    if new_range[0] == -1 or new_range[1] == -1:
        print(new_range)
        print(sloppy_tokens)
        print(word_range)
        raise Exception('range is incorrect')

    # print('------')
    # print(' '.join(sloppy_tokens))
    # print(' '.join(new_tokens))
    # print(word_range)
    # print(new_range)
    # print(' '.join(new_tokens[new_range[0] : new_range[1]]))
    # import pdb; pdb.set_trace()
    return new_tokens, new_range

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))



def get_tokenized_sent(sent):

    new_s = ''
    new_s_indices = []
    removed_s = ''
    removed_s_indices = []

    for ch_idx, ch in enumerate(sent):
        if ch == ' ' or ch in string.punctuation:
            removed_s += ch
            removed_s_indices.append(ch_idx)
        else:
            new_s += ch
            new_s_indices.append(ch_idx)
    return new_s, new_s_indices, removed_s, removed_s_indices


def find_longest_overlapping_region(overlaps, threshold):
    longest_num = 0
    longest_range = (0,0)
    cur_num = 0
    cur_range = (0,0)
    for idx, overlap in enumerate(overlaps):
        if overlap >= threshold:
            cur_num += 1
            cur_range = (cur_range[0], cur_range[1]+1)
        else:
            if cur_num > longest_num:
                longest_num = cur_num
                longest_range = cur_range
            cur_num = 0
            cur_range = (idx + 1, idx + 1)
    if cur_num > longest_num:
        longest_num = cur_num
        longest_range = cur_range
    return longest_range

def fix_hit(hit):
    for poc_idx in range(3):
        for line in hit:
            has_empty_field = False
            poc_type = line['Answer.coref-type %d' % poc_idx]
            if poc_type == '{}' or poc_type == '':
                has_empty_field = True
            for sent_idx in range(3):
                sel = line['Answer.sel-%d-%d' % (poc_idx, sent_idx)]
                if sel == '{}' or sel == '':
                    has_empty_field = True
            if has_empty_field:
                line['Answer.coref-type %d' % poc_idx] = ""
                for sent_idx in range(3):
                    line['Answer.sel-%d-%d' % (poc_idx, sent_idx)] = ""
    return hit

def fleiss_kappa(M):
    annotators = np.sum(M, axis=1, dtype=float)
    uniques = np.unique(annotators)
    fleiss_list = []
    num_in_list = []
    for num_ann in uniques:
        if num_ann == 1:
            continue
        which_rows = np.squeeze(np.argwhere(annotators == num_ann))
        my_M = M[which_rows]
        fleiss, p, P = fleiss_kappa_(my_M)
        num_in_list.append(len(which_rows))
        fleiss_list.append(fleiss)
        print(num_ann, len(which_rows), fleiss)
    sum_nums = sum(num_in_list)
    prob = [x * 1. / sum_nums for x in num_in_list]
    final_fleiss = sum([prob[i] * x for i, x in enumerate(fleiss_list)])
    return final_fleiss, p, P
    

def fleiss_kappa_(M):
  """
  See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
  :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
  :type M: numpy matrix
  """
  N, k = M.shape  # N is # of items, k is # of categories
  n_annotators = float(np.sum(M[0, :]))  # # of annotators
  P_annotators = np.sum(M, axis=1, dtype=float)
  p_annotators = np.sum(P_annotators)/N

  p = np.sum(M, axis=0) / (N * n_annotators)
  P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
  Pbar = np.sum(P) / N
  PbarE = np.sum(p * p)

  kappa = (Pbar - PbarE) / (1 - PbarE)

  return kappa, p, P

def fleiss_data_to_matrix(fleiss_data):
    num_assigns = len(list(set([x[1] for x in fleiss_data])))
    assign_id_to_idx = {}
    categories = 2
    M = np.zeros((num_assigns, categories))
    cur_idx = 0
    for x in fleiss_data:
        if x[1] not in assign_id_to_idx:
            assign_id_to_idx[x[1]] = cur_idx
            cur_idx += 1
        i = assign_id_to_idx[x[1]]
        j = int(x[2])
        M[i,j] = M[i,j]+1
    return M

def unfix_bracket_tokens_in_sent(sent):
    return sent.replace('-lrb-', '(').replace('-rrb-', ')').replace('-lsb-', '[').replace('-rsb-', ']').replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']')



# raw_csv_file = os.path.join('mturk','main_task','quality_check_results','Batch_3841416_batch_results.csv')
#
# all_lines = []
# with open(raw_csv_file, 'r' ) as f:
#     reader = csv.DictReader(f)
#     for line in reader:
#         all_lines.append(line)

raw_csv_folder = os.path.join('mturk','main_task','results','*.csv')

abstracts_test = [abs.strip().split('\t') for abs in open(os.path.join('stuff','abstracts_test.tsv'), encoding="utf8").read().strip().split('\n')]
abstracts_val = [abs.strip().split('\t') for abs in open(os.path.join('stuff','abstracts_val.tsv'), encoding="utf8").read().strip().split('\n')]
uppercased_abstracts = [abstracts_test, abstracts_val]

all_lines = []
for raw_csv_file in glob.glob(raw_csv_folder):
    with open(raw_csv_file, 'r' ) as f:
        reader = csv.DictReader(f)
        for line in reader:
            all_lines.append(line)


hits = {}
for line in all_lines:
    hitid = line['HITId']
    pair_idx = line['Input.pair_idx']
    if pair_idx not in hits:
        hits[pair_idx] = []
    hits[pair_idx].append(line)


assign_ids = []
data = {}
# for pair_idx, hit in tqdm(hits.items()):
#     if pair_idx == '1000011':
#         a=0
#     if len(hit) <= 1:
#         continue
#     for line in hit:
#         worker_id = line['WorkerId']
#         for sent_idx in range(3):
#             sent = line['Answer.source-text-0-%d' % sent_idx]
#             tokens = sent.split()
#             selected = [0] * len(tokens)
#             for poc_idx in range(3):
#                 beg = line['Answer.idx-beg-%d-%d' % (poc_idx, sent_idx)]
#                 if beg == '{}' or beg == '':
#                     continue
#                 beg = int(beg)
#                 end = int(line['Answer.idx-end-%d-%d' % (poc_idx, sent_idx)]) + 1
#                 selected[beg:end] = [1] * (end-beg)
#                 a=0
#             for token_idx, token in enumerate(tokens):
#                 assign_id = '%s_%d_%d' % (pair_idx, sent_idx, token_idx)
#                 val = selected[token_idx]
#                 fleiss_data.append((worker_id, assign_id, val))
#                 assign_ids.append(assign_id)
#                 if assign_id not in data:
#                     data[assign_id] = []
#                 data[assign_id].append(val)

# fleiss = AnnotationTask(fleiss_data).multi_kappa()
# print('Fleiss: %.4f' % fleiss)


fleiss_data = []
fleiss_data_tokens = []
poc_type_conflicts = 0
valid_pairs = 0
all_final_pocs = []
for pair_idx, hit in tqdm(hits.items()):
    my_fleiss_data = []
    # if pair_idx != '10001':
    #     continue
    # if poc_type_conflicts > 10:
    #     break
    # if valid_pairs > 2:
    #     break
    article_id = int(hit[0]['Input.article_id'])
    summ_sent_idx = int(hit[0]['Answer.summ_sent_idx'])
    # pair_idx = hit[0]['Input.pair_idx']
    sent_regions = []
    sent_worker_ids = []
    poc_types = []
    distance_matrix = None
    hit = fix_hit(hit)
    for line in hit:
        for poc_idx in range(3):
            poc_type = line['Answer.coref-type %d' % poc_idx]
            if poc_type == '{}' or poc_type == '':
                continue
            poc_types.append(poc_type)
    for sent_idx in range(3):
        regions = []
        worker_ids = []
        for line in hit:
            worker_id = line['WorkerId']
            for poc_idx in range(3):
                poc_type = line['Answer.coref-type %d' % poc_idx]
                sel = line['Answer.sel-%d-%d' % (poc_idx, sent_idx)]
                if sel == '{}' or sel == '':
                    continue
                regions.append(sel)
                worker_ids.append(worker_id)
        similarity_matrix = np.zeros((len(regions), len(regions)))
        for i in range(len(regions)):
            for j in range(len(regions)):
                r1 = regions[i].lower().split()
                r2 = regions[j].lower().split()
                rouge_l = calc_ROUGE_L_score(r1, r2)
                similarity_matrix[i,j] = rouge_l
        my_distance_matrix = 1 - similarity_matrix
        if distance_matrix is None:
            distance_matrix = my_distance_matrix
        else:
            distance_matrix += my_distance_matrix
        sent_regions.append(regions)
        sent_worker_ids.append(worker_ids)
    distance_matrix /= 3
    clusters = DBSCAN(min_samples=1, eps=1.1).fit_predict(distance_matrix)
    # print('\n\n\n')
    # for cluster_idx in range(max(clusters) + 1):
    #     my_regions = [[region for region_idx, region in enumerate(regions) if clusters[region_idx] == cluster_idx] for regions in sent_regions]
    #     for regions in my_regions:
    #         print(regions)
    #     print('----------')
    # print('\n')

    # print(clusters)
    # import pdb; pdb.set_trace();

    fleiss_overlaps = {}
    fleiss_overlaps_tokens = {}
    final_pocs = []    # List of tuples -- each tuple represents a PoC. (sent1, sent1_selection, sent1_selection_indices,
                                # sent2, sent2_selection, sent2_selection_indices, sent3, sent3_selection, sent3_selection_indices, PoC_type)
    for cluster_idx in range(max(clusters) + 1):
        my_sents_regions = [[region for region_idx, region in enumerate(regions) if clusters[region_idx] == cluster_idx] for regions in sent_regions]
        my_sents_worker_ids = [[worker_id for worker_id_idx, worker_id in enumerate(worker_ids) if clusters[worker_id_idx] == cluster_idx] for worker_ids in sent_worker_ids]
        my_poc_types = [poc_type for type_idx, poc_type in enumerate(poc_types) if clusters[type_idx] == cluster_idx]
        if len(my_poc_types) < 2:
            continue
        final_poc = []
        for sent_idx, regions in enumerate(my_sents_regions):
            worker_ids = my_sents_worker_ids[sent_idx]
            # print(regions)
            # sent_tokens = hit[0]['Answer.source-text-0-%d' % sent_idx].split()
            # sent_tokens = without_punctuation(hit[0]['Answer.source-text-0-%d' % sent_idx]).split()
            if sent_idx == 2:
                if article_id > 100000:
                    split = 1
                else:
                    split = 0
                article_idx = article_id % 100000
                abstracts = uppercased_abstracts[split]
                sent = unfix_bracket_tokens_in_sent(abstracts[article_idx][summ_sent_idx])
                sent_lower = sent.lower().replace('ñ', '').replace('š', 's')
                sent_nopunct, sent_nopunct_indices, sent_punct, sent_punct_indices = without_punctuation(sent_lower)

                sent_nopunct = sent_nopunct.lower()
                fake_sent = hit[0]['Answer.source-text-0-%d' % sent_idx]
                fake_sent_nopunct, fsent_nopunct_indices, fsent_punct, fsent_punct_indices = without_punctuation(fake_sent)
                if sent_nopunct != fake_sent_nopunct:
                    a=0
                a=0
            else:
                sent = hit[0]['Answer.source-text-0-%d' % sent_idx]
                sent_nopunct, sent_nopunct_indices, sent_punct, sent_punct_indices = without_punctuation(sent)
            char_idx_to_token_idx = []
            tokens = sent.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split()
            cur_idx = 0
            for token_idx, token in enumerate(tokens):
                char_idx_to_token_idx.extend([token_idx] * (len(token) + 1))
            # sent_tokens = [without_punctuation(token) for token in sent_tokens]
            overlaps = [0] * len(sent_nopunct)
            for region_idx, region in enumerate(regions):
                worker_id = worker_ids[region_idx]
                # region_tokens = region.split()
                # region_tokens = without_punctuation(region).split()
                region_nopunct, region_nopunct_indices, region_punct, region_punct_indices = without_punctuation(region)
                # region_tokens = [without_punctuation(token) for token in region_tokens]
                # match_start_idx, match_end_idx = find_sub_list(region_tokens, sent_tokens)
                # matches = subfinder(sent_tokens, region_tokens)
                matches = find_all(sent_nopunct, region_nopunct)
                if len(matches) == 0:
                    print('failed to find substring match')
                    # import pdb; pdb.set_trace();
                    a=0
                for match in matches:
                    # import pdb; pdb.set_trace();
                    for i in range(match, match + len(region_nopunct)):
                        overlaps[i] += 1
                if len(matches) > 0:
                    fleiss_overlaps_key = worker_id + '.' + str(sent_idx)
                    if fleiss_overlaps_key not in fleiss_overlaps:
                        fleiss_overlaps[fleiss_overlaps_key] = [0] * len(sent_nopunct)
                        fleiss_overlaps_tokens[fleiss_overlaps_key] = [0] * len(sent.split())
                    fleiss_overlaps[fleiss_overlaps_key][matches[0] : matches[0] + len(region_nopunct)] = [1] * len(region_nopunct)
                    token_start_idx = char_idx_to_token_idx[sent_nopunct_indices[matches[0]]]
                    try:
                        token_end_idx = char_idx_to_token_idx[sent_nopunct_indices[matches[0] + len(region_nopunct) -1]+1]
                    except:
                        print(matches[0], sent_nopunct_indices, char_idx_to_token_idx, len(region_nopunct), token_start_idx, sent_nopunct_indices[matches[0] + len(region_nopunct)-1]+1)
                    fleiss_overlaps_tokens[fleiss_overlaps_key][token_start_idx : token_end_idx] = [1] * (token_end_idx - token_start_idx)
            longest_range = find_longest_overlapping_region(overlaps, 2)
            if longest_range == (0,0):
                longest_range = find_longest_overlapping_region(overlaps, 1)
            if longest_range == (0,0):
                longest_range_fixed = (0,0)
            else:
                longest_range_fixed = (sent_nopunct_indices[longest_range[0]], sent_nopunct_indices[longest_range[1]-1]+1)
            # print('\n')
            # print(' '.join(sent_tokens[longest_range[0]: longest_range[1]]))
            # print(sent[longest_range_fixed[0] : longest_range_fixed[1]])

            sloppy_tokenized_sent = sent.split(' ')
            longest_range_word_level = character_to_word_level(sloppy_tokenized_sent, longest_range_fixed)
            # print(' '.join(sloppy_tokenized_sent[longest_range_word_level[0] : longest_range_word_level[1]]))

            if sent_idx == 2:
                fixed_tokenized_sent = sloppy_tokenized_sent
                fixed_range = longest_range_word_level
                fixed_tokenized_region = fixed_tokenized_sent[fixed_range[0] : fixed_range[1]]
                if ' '.join(fixed_tokenized_region) != sent[longest_range_fixed[0] : longest_range_fixed[1]]:
                    print(fixed_tokenized_region)
                    print(sent[longest_range_fixed[0] : longest_range_fixed[1]])
                    a=0
                a=0
            else:
                fixed_tokenized_sent, fixed_range = fix_punctuation(sloppy_tokenized_sent, longest_range_word_level)
                fixed_tokenized_region = fixed_tokenized_sent[fixed_range[0] : fixed_range[1]]
            # import pdb; pdb.set_trace();
            final_poc.extend([' '.join(fixed_tokenized_sent), ' '.join(fixed_tokenized_region), ' '.join([str(i) for i in fixed_range])])
        # print(my_poc_types)
        counter = Counter(my_poc_types)
        most_common = counter.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            poc_type_conflicts += 1
        final_poc_type = most_common[0][0]
        # print(final_poc_type)
        final_poc.append(final_poc_type)
        final_pocs.append(final_poc)
        # print('----------')
    if len(final_pocs) > 0:
        valid_pairs += 1
        final_pocs = [[str(pair_idx), str(article_id)] + poc for poc in final_pocs]
        all_final_pocs.append(final_pocs)

        cids = {}
        cur_cid = -1
        for fleiss_overlaps_key, overlaps in fleiss_overlaps.items():
            for token_idx, val in enumerate(overlaps):
                worker_id, sent_idx = fleiss_overlaps_key.split('.')
                if worker_id not in cids:
                    cur_cid += 1
                    cids[worker_id] = str(cur_cid)
                cid = cids[worker_id]
                assign_id = str(pair_idx) + '.' + str(sent_idx) + '.' + str(token_idx)
                item = (cid, assign_id, val)
                my_fleiss_data.append(item)
                fleiss_data.append(item)
        for fleiss_overlaps_key, overlaps in fleiss_overlaps_tokens.items():
            for token_idx, val in enumerate(overlaps):
                worker_id, sent_idx = fleiss_overlaps_key.split('.')
                if worker_id not in cids:
                    cur_cid += 1
                    cids[worker_id] = str(cur_cid)
                cid = cids[worker_id]
                assign_id = str(pair_idx) + '.' + str(sent_idx) + '.' + str(token_idx)
                item = (cid, assign_id, val)
                # my_fleiss_data_tokens.append(item)
                fleiss_data_tokens.append(item)
    else:
        item = [str(pair_idx), str(article_id)]
        for sent_idx in range(3):
            if sent_idx == 2:
                if article_id > 100000:
                    split = 1
                else:
                    split = 0
                article_idx = article_id % 100000
                abstracts = uppercased_abstracts[split]
                sent = unfix_bracket_tokens_in_sent(abstracts[article_idx][summ_sent_idx])
            else:
                sent = hit[0]['Answer.source-text-0-%d' % sent_idx]
            sloppy_tokenized_sent = sent.split(' ')
            fixed_tokenized_sent, fixed_range = fix_punctuation(sloppy_tokenized_sent, (0,0))
            item.extend([' '.join(fixed_tokenized_sent), '', ''])
        item.append('')
        all_final_pocs.append([item])
    # print('\n')

    # if len(my_fleiss_data) > 0:
    #     # fleiss = AnnotationTask(my_fleiss_data).multi_kappa()
    #     M = fleiss_data_to_matrix(my_fleiss_data)
    #     fleiss, p, P = fleiss_kappa(M)
    #     print('Fleiss: %.4f' % fleiss)


# fleiss = AnnotationTask(fleiss_data).multi_kappa()
M = fleiss_data_to_matrix(fleiss_data)
fleiss, p, P = fleiss_kappa(M)
print('Fleiss: %.4f' % fleiss)

M = fleiss_data_to_matrix(fleiss_data_tokens)
fleiss, p, P = fleiss_kappa(M)
print('Fleiss tokens: %.4f' % fleiss)

# with open(out_file, "w") as f:
#     f.write('\t'.join(['pair_id', 'article_id', 'sent1', 'sent1_selection', 'sent1_selection_indices', 'sent2', 'sent2_selection', 'sent2_selection_indices', 'sent3', 'sent3_selection', 'sent3_selection_indices', 'PoC_type']) + '\n\n')
#     for pocs in all_final_pocs:
#         if len(pocs) == 0:
#             f.write('\n')
#         else:
#             for poc in pocs:
#                 f.write('\t'.join(poc) + '\n')
#         f.write('\n')


print('done')


    

    




            















