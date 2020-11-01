import os
import json
from tqdm import tqdm
import csv

def fix_tokenization(sent):
    '''Necessary to fix slight tokenization inconsistencies in the article'''
    sent = ' '.join([token[:-1] + ' .' if token[-1] == '.' else token for token in sent.split(' ')])
    sent = sent.replace('..', '. .')
    return sent

pair_idx_to_fusion = {}
with open('mturk/processed/valid_fusions.csv') as f:
    reader = csv.reader(f)
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        pair_idx_to_fusion[int(row[0])] = row

id_to_article = {}
for dataset_split in ['val', 'test']:
    with open('data/processed/cnn_dm/' + dataset_split + '/articles.tsv', encoding="utf8") as f:
        articles = f.read().strip().split('\n')
    for idx, article in enumerate(articles):
        article_sent_tokens = [fix_tokenization(sent.strip()).split(' ') for sent in article.split('\t')]
        fixed_article = '\t'.join([' '.join(sent) for sent in article_sent_tokens])
        if dataset_split == 'val':
            idx += 1000000
        id_to_article[idx] = fixed_article
id_to_summary = {}
for dataset_split in ['val', 'test']:
    with open('data/processed/cnn_dm/' + dataset_split + '/summaries.tsv', encoding="utf8") as f:
        articles = f.read().strip().split('\n')
    for idx, article in enumerate(articles):
        if dataset_split == 'val':
            idx += 1000000
        id_to_summary[idx] = article

with open('mturk/main_task/processed/PoC.tsv') as f:
    text = f.read()

poc_type_dict = {
    'Name': 'Nominal',
    'Pronoun': 'Pronominal',
    'Non-Name': 'Common-Noun',
    'Repetition': 'Repetition',
    'Event': 'Event',
}

instances = text.split('\n\n')
instances = instances[1:]
if instances[-1].strip() == '':
    instances = instances[:-1]
fusion_list = []
for instance in tqdm(instances):
    # print(instance)
    fusion = {}
    pocs = instance.split('\n')
    fusion_poc_list = []
    for poc in pocs:
        if poc == '':
            continue
        # print(poc.split('\t'))
        pair_id, article_id, sent1, sent1_selection, sent1_selection_indices, sent2, sent2_selection, sent2_selection_indices, sent3, sent3_selection, sent3_selection_indices, PoC_type = poc.split('\t')
        pair_id = int(pair_id)
        fusion['Example_Id'] = int(pair_id)
        # fusion['Article_Id'] = int(article_id)
        fusion['Sentence_1'] = sent1
        fusion['Sentence_2'] = sent2
        fusion['Sentence_Fused'] = sent3
        article_id = int(article_id)
        extra_info = pair_idx_to_fusion[pair_id]
        fusion['Sentence_1_Index'] = int(extra_info[3].split(',')[0])
        fusion['Sentence_2_Index'] = int(extra_info[3].split(',')[1])
        fusion['Sentence_Fused_Index'] = int(extra_info[5])

        fusion['Full_Article'] = id_to_article[article_id]
        fusion['Full_Summary'] = id_to_summary[article_id]
        if sent1_selection == '':
            continue
        out_poc = {
            'Sentence_1_Selection': [int(idx) for idx in sent1_selection_indices.split()],
            'Sentence_2_Selection': [int(idx) for idx in sent2_selection_indices.split()],
            'Sentence_Fused_Selection': [int(idx) for idx in sent3_selection_indices.split()],
            'PoC_Type': poc_type_dict[PoC_type]
        }
        fusion_poc_list.append(out_poc)
    fusion['PoCs'] = fusion_poc_list
    fusion_list.append(fusion)
fusion_list.insert(0, fusion_list.pop(14))
fusion_list.insert(0, fusion_list.pop(23))
fusion_list.insert(0, fusion_list.pop(42))
fusion_list.insert(1, fusion_list.pop(6))

with open('mturk/main_task/processed/PoC_dataset.json', 'w') as f:
    json.dump(fusion_list, f, indent=4)