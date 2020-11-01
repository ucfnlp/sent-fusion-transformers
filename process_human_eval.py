
from tqdm import tqdm

import csv
import glob
import os
import numpy as np

csv_dir = os.path.join('mturk','model_eval','raw_results')
# valid_fusions_file = os.path.join('mturk','model_eval','processed','valid_fusions.csv')
all_fusions_file = os.path.join('mturk','all_pairs_test_and_val.csv')


models = ['pg', 'transformer', 'coref_head', 'reference']
model_to_idx = {}
for model_idx, model in enumerate(models):
    model_to_idx[model] = model_idx
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

all_lines = []
for csv_file in csv_files:
    with open(csv_file, 'r' ) as f:
        reader = csv.DictReader(f)
        for line in reader:
            all_lines.append(line)

hits = {}
for line in all_lines:
    hitid = line['HITId']
    if hitid not in hits:
        hits[hitid] = []
    hits[hitid].append(line)

valid_fusions = []
invalid_fusions = []
rejected = 0
failed_quality = 0
voted_often = 0
total = 0
model_results = {}
for model_idx, model in enumerate(models):
    model_results[model] = []
for hitid, assigns in hits.items():
    num_yes = 0
    num_no = 0
    my_results = {}
    for assign in assigns:
        reference_labelled_true = True
        for cur_idx in range(len(models)):
            total += 1
            model = assign['Answer.model_' + str(cur_idx)]
            if assign['Answer.Faithfulness_' + str(cur_idx)] == 'YES':
                my_results[model] = 1
            else:
                my_results[model] = 0
                if model == 'reference':
                    reference_labelled_true = False
        if True:
        # if reference_labelled_true:
            for model in models:
                model_results[model].append(my_results[model])


print('Percentage faithful to the original: ')
for model_idx, model in enumerate(models):
    print('%s\t\t%.2f' % (model, np.mean(model_results[model])*100))