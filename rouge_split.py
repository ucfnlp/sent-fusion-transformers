import os
import rouge_functions

input_dir = 'data/correctness/processed'
output_dir = 'data/correctness/rouge'

# systems = ['reference']
systems = ['reference', 'bert-abs', 'bert-extr', 'dca', 'pg', 'bottom-up']
rouge_splits = ['all', 'singletons', 'pairs']

system = 'reference'
l_param=100

reference_indices = {
    'all': [],
    'singletons': [],
    'pairs': []
}

for system in systems:
    print(system)
    system_dir = os.path.join(input_dir, system)
    with open(os.path.join(system_dir, 'summaries_tokenized.txt')) as f:
        text = f.read().strip()
    summary_sents_all = [summ.split('\t') for summ in text.split('\n')]
    with open(os.path.join(system_dir, 'source_indices.txt')) as f:
        text = f.read().strip()
    ssi_all = [[source_indices.split(',') for source_indices in similar_source_indices.split('\t')] for similar_source_indices in text.split('\n')]
    ssi_all = [[[] if source_indices == [-1] else source_indices for source_indices in similar_source_indices] for similar_source_indices in ssi_all]
    for article_idx, summ in enumerate(summary_sents_all):
        ssi = ssi_all[article_idx]
        my_rouge_splits = rouge_splits
        for split in my_rouge_splits:
            rouge_dir = os.path.join(output_dir, system, split)
            if not os.path.exists(rouge_dir):
                os.makedirs(rouge_dir)
            if split == 'all' or system != 'reference':
                out_summ_sents = summ
            elif split == 'singletons':
                out_summ_sents = [sent for sent_idx, sent in enumerate(summ) if len(ssi[sent_idx]) == 1]
            elif split == 'pairs':
                out_summ_sents = [sent for sent_idx, sent in enumerate(summ) if len(ssi[sent_idx]) == 2]
            if len(out_summ_sents) == 0:
                continue
            if system == 'reference':
                reference_indices[split].append(article_idx)
            else:
                if article_idx not in reference_indices[split]:
                    continue
            out_text = '\n'.join(out_summ_sents)
            if system == 'reference':
                out_file_name = os.path.join(rouge_dir, "%06d_reference.%s.txt" % (
                    article_idx, chr(ord('A') + 0)))
            else:
                out_file_name = os.path.join(rouge_dir, "%06d_decoded.txt" % article_idx)
            with open(out_file_name, 'w') as f:
                f.write(out_text)

sheets_strss = []
for system in systems[1:]:
    sheets_strs = []
    for split in rouge_splits:
        print(system + ' ' + split)
        results_dir = os.path.join(output_dir, system, 'results', split)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_dict = rouge_functions.rouge_eval(os.path.join(output_dir, 'reference', split), os.path.join(output_dir, system, split), l_param=l_param)
        sheets_str = rouge_functions.rouge_log(results_dict, results_dir)
        sheets_strs.append(sheets_str.strip())
    sheets_strss.append(sheets_strs)

for sheets_strs in  sheets_strss:
    print('\t'.join(sheets_strs))
