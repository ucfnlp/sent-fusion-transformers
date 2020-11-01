import os
import csv
import numpy as np
import operator
import math
import glob

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

pair_indices = '''1000515
1000552
1001099
1000017
1000362
1000987
1000586
1000660
1000906
1001035
'''.strip().split('\n')


correct_responses_file = os.path.join('mturk','main_task','correct_responses.tsv')
with open(correct_responses_file) as f:
    text = f.read()
examples = text.split('\n\n')
correct_responses = {}
for example_idx, example in enumerate(examples):
    pocs = example.split('\n')
    ex = []
    for poc in pocs:
        items = [item.strip() for item in poc.strip().split('\t')]
        if len(items) != 4:
            print(example_idx, example)
            raise Exception('')
        ex.append(items)
    correct_responses[pair_indices[example_idx]] = ex

# raw_csv_file = os.path.join('mturk','main_task','quality_check_results','Batch_3841416_batch_results.csv')
#
# all_lines = []
# with open(raw_csv_file, 'r' ) as f:
#     reader = csv.DictReader(f)
#     for line in reader:
#         all_lines.append(line)

raw_csv_folder = os.path.join('mturk','main_task','quality_check_results2','*.csv')

all_lines = []
for raw_csv_file in glob.glob(raw_csv_folder):
    with open(raw_csv_file, 'r' ) as f:
        reader = csv.DictReader(f)
        for line in reader:
            all_lines.append(line)
worker_scores = {}
worker_worktimes = {}
hits = {}
for line in all_lines:
    hitid = line['HITId']
    if hitid not in hits:
        hits[hitid] = 0
    hits[hitid] += 1
    if line['WorkerId'] not in worker_scores:
        worker_scores[line['WorkerId']] = []
        worker_worktimes[line['WorkerId']] = []
    worker_worktimes[line['WorkerId']].append(int(line['WorkTimeInSeconds']))
    pair_idx = line['Input.pair_idx']
    my_correct_responses = correct_responses[pair_idx]
    
    rouge_scores = []
    type_scores = []
    num_ann_pocs = 0
    for poc_idx in range(3):
        sent_rouge_scores = []
        for sent_idx in range(3):
            sel = line['Answer.sel-%d-%d' % (poc_idx, sent_idx)]
            if sel == '{}' or sel == '':
                continue
            match_scores = []
            for corr_idx, corr in enumerate(my_correct_responses):
                rouge_l = calc_ROUGE_L_score(sel.lower().split(), corr[sent_idx].lower().split())
                match_scores.append(rouge_l)
            sent_rouge_scores.append(match_scores)
        if len(sent_rouge_scores) == 0:
            continue
        num_ann_pocs += 1
        sent_rouge_scores = np.array(sent_rouge_scores)
        sum_rouge_scores = np.mean(sent_rouge_scores, axis=0)

        best_corr_idx = np.argmax(sum_rouge_scores)
        best_rouge_score = np.max(sum_rouge_scores)
        rouge_scores.append(best_rouge_score)

        ann_type = line['Answer.coref-type %d' % poc_idx]
        corr_type = my_correct_responses[best_corr_idx][3]
        if ann_type == corr_type:
            type_scores.append(1)
        else:
            type_scores.append(0)

    final_rouge_score = np.mean(rouge_scores)
    final_type_score = np.mean(type_scores)

    if num_ann_pocs >= len(my_correct_responses):
        num_pocs_score = 1
    else:
        num_pocs_score = 0

    aggregate_score = final_rouge_score*2 + final_type_score + num_pocs_score
    print(final_rouge_score, final_type_score, num_pocs_score)
    worker_scores[line['WorkerId']].append(aggregate_score)

for workerid, worker_score in list(worker_scores.items()):
    if math.isnan(np.mean(worker_score)):
        del worker_scores[workerid]

sorted_workers = sorted(worker_scores, key=lambda k: sum(worker_scores[k]) / len(worker_scores[k]))
for workerid in sorted_workers:
    worker_score = worker_scores[workerid]
    if len(worker_score) >= 8 and np.mean(worker_score) >= 2:
        print(workerid, np.mean(worker_score), len(worker_score))
        # print(workerid)

for hitid, num_done in hits.items():
    if num_done != 10:
        print(hitid, num_done)

print('done')


    

    




            















