import glob
from tqdm import tqdm
import os
import numpy as np
import nltk

logs_dir = 'logs'
exp_name = 'cnn_dm__bert_both_pocd_pocgoldunilm_coref_l11_h0_fc_fm_summ100.0_ninst4'
exp_names = sorted([o for o in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir,o)) and 'cnn_dm__bert_both_crdunilm_coref' in o and 'triples' not in o])
for exp_name in tqdm(exp_names):
    decode_dirs = os.listdir(os.path.join(logs_dir, exp_name))
    for decode_dir in decode_dirs:
        # print(exp_name, decode_dir)
        sheets_results_file = os.path.join(logs_dir, exp_name, decode_dir, 'bert_sheets_results.txt')
        dec_dir = os.path.join(logs_dir, exp_name, decode_dir, 'decoded')
        ref_dir = os.path.join(logs_dir, exp_name, decode_dir, 'reference')
        if not os.path.exists(sheets_results_file):
            print('sheets file doesnt exist')
            print(sheets_results_file)
            continue
        with open(sheets_results_file) as f:
            sheets_results_contents = f.read().strip()
        print(sheets_results_contents)