import bert_score
import glob
from tqdm import tqdm
import os
import numpy as np
import nltk
import util
import data
import ssi_functions

logs_dir = 'logs'

exp_name = 'cnn_dm__bert_both_crdunilm_coref_l1_h0_fc_fm_triples_summ100.0_ninst4'
exp_names = [o for o in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir,o))]
for exp_name in tqdm(exp_names):
    if 'pocd' in exp_name:
        names_to_types = [('raw_article_sents', 'string_list'), ('summary_text', 'string'), ('coref_chains', 'delimited_list_of_list_of_lists')]
        data_root = os.path.expanduser('~') + '/data/tf_data/poc_fusions/cnn_dm/test*'
    elif 'triples' in exp_name:
        continue
        names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'),
                          ('summary_text', 'string'),
                          ('article_lcs_paths_list', 'delimited_list_of_list_of_lists'),
                          ('coref_chains', 'delimited_list_of_list_of_lists')]
        data_root = os.path.expanduser(
            '~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens_spacy_summinc_triples_fusions/cnn_dm/test*'
    else:
        names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'),
                          ('summary_text', 'string'),
                          ('article_lcs_paths_list', 'delimited_list_of_list_of_lists'),
                          ('coref_chains', 'delimited_list_of_list_of_lists')]
        data_root = os.path.expanduser(
            '~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens_spacy_fusions/cnn_dm/test*'
    decode_dirs = os.listdir(os.path.join(logs_dir, exp_name))
    for decode_dir in decode_dirs:
        new_sheets_results_file = os.path.join(logs_dir, exp_name, decode_dir, 'fusion_sheets_results.txt')
        if os.path.exists(new_sheets_results_file):
            continue
        print(exp_name, decode_dir)
        sheets_results_file = os.path.join(logs_dir, exp_name, decode_dir, 'bert_sheets_results.txt')
        dec_dir = os.path.join(logs_dir, exp_name, decode_dir, 'decoded')
        ref_dir = os.path.join(logs_dir, exp_name, decode_dir, 'reference')
        if not os.path.exists(sheets_results_file):
            print('sheets file doesnt exist')
            print(sheets_results_file)
            continue
        with open(sheets_results_file) as f:
            sheets_results_contents = f.read().strip().split('\t')
        # if len(sheets_results_contents) != 9:
        #     print('skipping because it is not just rouge scores in the results file', exp_name)
        #     continue

        decoded_files = sorted(glob.glob(os.path.join(dec_dir, '*')))
        reference_files = sorted(glob.glob(os.path.join(ref_dir, '*')))
        if len(decoded_files) != len(reference_files):
            print('skipping because len(decoded_files) != len(reference_files)', exp_name)
            continue
        cands = []
        refs = []
        lens = []
        is_fusion_list = []
        example_generator = data.example_generator(data_root, True, False,
                                               should_check_valid=False)
        for file in decoded_files:
            with open(file) as f:
                text = f.read().replace('\n', ' ')
                cands.append(text)
                lens.append(len(text.strip().split()))
            ex = next(example_generator)
            if 'pocd' in exp_name:
                raw_article_sents, groundtruth_summary_text, original_coref_chains = util.unpack_tf_example(ex, names_to_types)
            elif 'triples' in exp_name:
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, groundtruth_article_lcs_paths_list, original_coref_chains = util.unpack_tf_example(
                    ex, names_to_types)
            else:
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, groundtruth_article_lcs_paths_list, original_coref_chains = util.unpack_tf_example(
                    ex, names_to_types)
            decoded_words = text.split()
            min_matched_tokens = 2
            selected_article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
            highlight_summary_sent_tokens = [decoded_words]
            highlight_ssi_list, lcs_paths_list, highlight_article_lcs_paths_list, highlight_smooth_article_lcs_paths_list = ssi_functions.get_simple_source_indices_list(
                highlight_summary_sent_tokens,
                selected_article_sent_tokens, 2, min_matched_tokens)

            if len(highlight_ssi_list[0]) < 2:
                is_fusion_list.append(0)
            else:
                is_fusion_list.append(1)
        for file in reference_files:
            with open(file) as f:
                refs.append(f.read().replace('\n', ' '))
        # print('Calculating bert score')
        bleu = nltk.translate.bleu_score.corpus_bleu([[ref] for ref in refs], cands)*100
        # print(bleu)
        bert_p, bert_r, bert_f = bert_score.score(cands, refs, lang="en", verbose=False, batch_size=8, model_type='bert-base-uncased')
        # print(bert_p, bert_r, bert_f)
        avg_len = np.mean(lens)
        bert_p = np.mean(bert_p.cpu().numpy())*100
        bert_r = np.mean(bert_r.cpu().numpy())*100
        bert_f = np.mean(bert_f.cpu().numpy())*100
        # print(bert_p)

        percent_fusion = np.mean(is_fusion_list) * 100
        new_results = ['%.2f'%percent_fusion] + sheets_results_contents
        print('\t'.join(new_results))
        new_sheets_results_file = os.path.join(logs_dir, exp_name, decode_dir, 'fusion_sheets_results.txt')
        with open(new_sheets_results_file, 'w') as f:
            f.write('\t'.join(new_results))
    # break