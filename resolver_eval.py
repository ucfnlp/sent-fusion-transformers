import itertools
import numpy as np
from tqdm import tqdm
import os
from allennlp import pretrained
import nltk
import json
import spacy
from spacy.tokens import Doc
import neuralcoref
# from stanfordnlp.server import CoreNLPClient
from stanfordcorenlp import StanfordCoreNLP
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import re

os.environ['CORENLP_HOME'] = '/home/logan/stanford-corenlp-full-2018-02-27'

# allen_model = pretrained.neural_coreference_resolution_lee_2017()
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")

nlp = spacy.load('en_core_web_sm')
def custom_tokenizer(text):
    tokens = text.split()
    doc = Doc(nlp.vocab, tokens)
    return doc
def add_sent_starts(doc):
    sent_start_indices = doc.user_data['sent_start_indices']
    for token_idx, token in enumerate(doc):
        if token_idx in sent_start_indices:
            token.is_sent_start = True
        else:
            token.is_sent_start = False
    return doc
nlp.tokenizer = custom_tokenizer
nlp.add_pipe(add_sent_starts, before="parser")
neuralcoref.add_to_pipe(nlp)

# # set up the client
# client = CoreNLPClient(properties={'annotators': ','.join(['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']), 'coref.algorithm' : 'statistical'}, timeout=60000, memory='16G')

snlp = StanfordCoreNLP(r'/home/logan/stanford-corenlp-full-2018-02-27', quiet=False)
props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
myprops = {'annotators': 'coref', 'pipelineLanguage': 'en', 'tokenize.whitespace': 'true', 'ssplit.eolonly': 'true'}

def process_sent(sent, whitespace=False):
    # line = decode_text(sent.lower())
    line = sent
    if whitespace:
        tokenized_sent = line.split(' ')
    else:
        tokenized_sent = nltk.word_tokenize(line)
    return tokenized_sent

def allen(tokens, article_sent_tokens):

    def get_sent_and_word_idx_of_mention(mention, article_sent_tokens):
        num_tokens_before = 0
        cur_idx = 0
        abs_start, abs_end = mention
        abs_end += 1
        # import pdb; pdb.set_trace()
        for sent_idx, sent in enumerate(article_sent_tokens):
            cur_idx += len(sent)
            if abs_start < cur_idx:
                start_idx = abs_start - num_tokens_before
                end_idx = abs_end - num_tokens_before
                return sent_idx, start_idx, end_idx
            num_tokens_before += len(sent)
        print(abs_start)
        print(abs_end)
        print(article_sent_tokens)
        raise Exception('Mention token idx larger than number of tokens in article.')

    # print(tokens)
    # print('----------------------------------')
    # results = allen_model.predict_tokenized(tokens)
    results = predictor.predict(
        document=' '.join(tokens)
    )
    # print('----------------------------------')
    # print(results)
    wrong_idx_to_real_idx = {}
    if len(results['document']) != len(tokens):
        new_doc = []
        max_len = max(len(results['document']), len(tokens))
        i = 0
        j = 0
        while i < len(results['document']) and j < len(tokens):
            # print(results['document'][i] + '\t' + tokens[j])
            if results['document'][i] == tokens[j]:
                new_doc.append(results['document'][i])
            else:
                built_token = results['document'][i]
                if tokens[j].startswith(built_token):
                    while i < len(results['document']) and built_token != tokens[j]:
                        wrong_idx_to_real_idx[i] = j
                        # print(built_token)
                        i += 1
                        built_token += results['document'][i]
                    if built_token == tokens[j]:
                        # print(built_token)
                        new_doc.append(built_token)
                    else:
                        print(built_token)
                        print(results['document'])
                        print(tokens)
                        raise Exception('could not build token')
                else:
                    new_doc.append(results['document'][i])
            wrong_idx_to_real_idx[i] = j
            i += 1
            j += 1

        min_len = min(len(new_doc), len(tokens))
        if len(new_doc) != len(tokens):
            print(len(new_doc))
            print(len(tokens))
            for i in range(min_len):
                print(new_doc[i] + '\t' + tokens[i])
            raise Exception('''len(results['document']) != len(tokens)''')
    else:
        for i in range(len(tokens)):
            wrong_idx_to_real_idx[i] = i
    # print(wrong_idx_to_real_idx)
    cluster_list = []
    for cluster in results['clusters']:
        mention_list = []
        for mention_idx, mention in enumerate(cluster):
            fixed_mention = [wrong_idx_to_real_idx[mention[0]], wrong_idx_to_real_idx[mention[1]]]
            sent_idx, start_idx, end_idx = get_sent_and_word_idx_of_mention(fixed_mention, article_sent_tokens)
            mention_text = ' '.join(article_sent_tokens[sent_idx][start_idx:end_idx])
            if mention_text != ' '.join(results['document'][mention[0] : mention[1] + 1]) and mention_text.replace(' ','') != ' '.join(results['document'][mention[0] : mention[1] + 1]).replace(' ',''):
                # print(mention_text)
                # print(' '.join(results['document'][mention[0] : mention[1] + 1]))
                # print('**********')
                # import pdb; pdb.set_trace()
                # raise Exception('Mention was incorrectly found')
                a=0

            mention_dict = {
                'endIndex': end_idx + 1,
                'startIndex': start_idx + 1,
                'text': mention_text,
                'sentNum': sent_idx + 1,
                # 'isRepresentativeMention': mention == rep,
            }
            mention_list.append(mention_dict)
            a=0
        cluster_list.append(mention_list)
        # print('')
    # import pdb; pdb.set_trace()
    return cluster_list

def spacy_coref(tokens, article_sent_tokens):
    def get_sent_and_word_idx_of_mention(mention, article_sent_tokens):
        num_tokens_before = 0
        cur_idx = 0
        for sent_idx, sent in enumerate(article_sent_tokens):
            cur_idx += len(sent)
            if mention.start < cur_idx:
                if mention.text in ' '.join(sent):
                    start_idx = mention.start - num_tokens_before
                    end_idx = mention.end - num_tokens_before
                    return sent_idx, start_idx, end_idx
                else:
                    print(mention.text.lower())
                    print(mention.start)
                    print(mention.end)
                    print(mention.sent)
                    print(' '.join(sent))
                    print(article_sent_tokens)
                    raise Exception('Mention token idx was in sent, but mention text could not be found in the sent.')
            num_tokens_before += len(sent)
        print(mention.text.lower())
        print(mention.start)
        print(mention.end)
        print(mention.sent)
        print(article_sent_tokens)
        raise Exception('Mention token idx larger than number of tokens in article.')

    def get_sent_start_indices(article_sent_tokens):
        indices = []
        cur_idx = 0
        for sent in article_sent_tokens:
            indices.append(cur_idx)
            cur_idx += len(sent)
        return indices

    all_sent_tokens = article_sent_tokens
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
        for mention in cluster.mentions:
            sent_idx, start_idx, end_idx = get_sent_and_word_idx_of_mention(mention, all_sent_tokens)
            print(mention.text)
            print(' '.join(tokens[mention.start : mention.end]))
            print('**********')
            mention_dict = {
                'endIndex': end_idx+1,
                'startIndex': start_idx+1,
                'text': mention.text,
                'sentNum': sent_idx+1,
                # 'isRepresentativeMention': mention == cluster.main,
            }
            mention_list.append(mention_dict)
        cluster_list.append(mention_list)
    return cluster_list

def stanford(tokens, article_sent_tokens):

    # text = 'Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008.'
    # text = 'Barack was born in Hawaii.\nHis wife Michelle was born in Milan.\nHe says that she is very smart.'
    text = '\n'.join([' '.join(sent) for sent in article_sent_tokens])

    # result = json.loads(snlp.annotate(text, properties=props))
    result = json.loads(snlp.annotate(text, properties=myprops))

    clusters = list(result['corefs'].items())
    for cluster in clusters:
        num, mentions = cluster
        for mention in mentions:
            print(mention)

    cluster_list = []
    for cluster in clusters:
        num, mentions = cluster
        mention_list = []
        for mention in mentions:
            sent_idx, start_idx, end_idx = mention['sentNum']-1, mention['startIndex']-1, mention['endIndex']-1
            print(' '.join(article_sent_tokens[sent_idx][start_idx : end_idx]))
            print(mention['text'])
            print('**********')
            mention_dict = {
                'endIndex': end_idx+1,
                'startIndex': start_idx+1,
                'text': mention['text'],
                'sentNum': sent_idx+1,
                # 'isRepresentativeMention': mention == cluster.main,
            }
            mention_list.append(mention_dict)
        cluster_list.append(mention_list)
    return cluster_list


in_folder = os.path.join('mturk','main_task','processed')
in_file = os.path.join(in_folder, 'PoC.tsv')

with open(in_file, encoding = "ISO-8859-1") as f:
    raw_fusions = f.read().strip().split('\n\n')[1:]

fusions = []
for raw_pocs in raw_fusions:
    if raw_pocs == '':
        fusions.append([])
    else:
        pocs = [raw_poc.split('\t') for raw_poc in raw_pocs.split('\n')]
        if pocs[0][-1] == '':
            pocs = [[int(poc[0]), int(poc[1]), poc[2], poc[3], [int(i) for i in poc[4].split()],
                poc[5], poc[6], [int(i) for i in poc[7].split()],
                poc[8], poc[9], [int(i) for i in poc[10].split()], poc[11]] for poc in pocs]
            fusions.append(pocs)
            continue
        pocs = [[int(poc[0]), int(poc[1]), poc[2], poc[3], [int(i) for i in poc[4].split()],
                poc[5], poc[6], [int(i) for i in poc[7].split()],
                poc[8], poc[9], [int(i) for i in poc[10].split()], poc[11]] for poc in pocs]
        fusions.append(pocs)

preprocess = False

if preprocess:
    resolver = 'spacy'
    poc_types = []
    num_pocs = []
    article_ids = []
    cluster_lists = []
    for fusion_idx, pocs in enumerate(tqdm(fusions)):
        poc = pocs[0]
        pair_id = poc[0]
        article_id = poc[1]
        sent1 = poc[2]
        sent1_selection = poc[3]
        sent1_selection_indices = poc[4]
        sent2 = poc[5]
        sent2_selection = poc[6]
        sent2_selection_indices = poc[7]
        sent3 = poc[8]
        sent3_selection = poc[9]
        sent3_selection_indices = poc[10]
        poc_type = poc[11]

        raw_article_sents = [sent1, sent2]
        article_sent_tokens = [process_sent(sent, whitespace=True) for sent in raw_article_sents]
        text = ' '.join(raw_article_sents)
        tokens = text.split(' ')
        # import pdb; pdb.set_trace()
        if resolver == 'allen':
            cluster_list = allen(tokens, article_sent_tokens)
        elif resolver == 'spacy':
            cluster_list = spacy_coref(tokens, article_sent_tokens)
        else:
            cluster_list = stanford(tokens, article_sent_tokens)
        cluster_lists.append(cluster_list)
        a=0

    if not os.path.exists('mturk/corefs'):
        os.makedirs('mturk/corefs')
    with open('mturk/corefs/%s_corefs.json' % resolver, 'w', encoding='utf-8') as f:
        json.dump(cluster_lists, f, ensure_ascii=False, indent=4)


def word_tag_eval(all_gt_word_tags, all_sys_word_tags):
    result = precision_recall_fscore_support(all_gt_word_tags, all_sys_word_tags)
    suffix = '\t'.join(str(score) for score in result) + '\n'
    return suffix

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


class Pmf(Counter):
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __add__(self, other):
        """Adds two distributions.

        The result is the distribution of sums of values from the
        two distributions.

        other: Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                pmf[key1 + key2] += prob1 * prob2
        return pmf

    def __hash__(self):
        """Returns an integer hash value."""
        return id(self)

    def __eq__(self, other):
        return self is other

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))

def cluster_in_both_sents(cluster):
    found_mention = [False, False]
    for mention in cluster:
        found_mention[mention['sentNum']-1] = True
    return found_mention[0] and found_mention[1]

def only_first_in_each_sent(cluster):
    new_cluster = [None, None]
    start_indices = [100000, 100000]
    for mention in cluster:
        sent_idx = mention['sentNum']-1
        start_idx = mention['startIndex']-1
        if start_idx < start_indices[sent_idx]:
            new_cluster[sent_idx] = mention
            start_indices[sent_idx] = start_idx
    return new_cluster

def only_numbers(s):
    nums = [ float(dec)*100 for dec in re.findall("\d+\.\d+", s)]
    return nums

def print_scores(pretty_resolver, scores):
    print('%s    &    %.1f    &    %.1f    &    %.1f' % tuple([pretty_resolver] + scores))

def print_recalls(pretty_resolver, recalls):
    print('%s    &    %.1f    &    %.1f    &    %.1f    &    %.1f    &    %.1f' % tuple([pretty_resolver] + recalls))

poc_unique_types = ['Pronoun', 'Name', 'Non-Name', 'Repetition', 'Event']
resolvers = ['spacy', 'allen', 'stanford']
pretty_resolvers = ['SpaCy', 'AllenNLP', 'Stanford']
print(poc_unique_types)
print(resolvers)
for resolver_idx, resolver in enumerate(resolvers):
    pretty_resolver = pretty_resolvers[resolver_idx]
    with open('mturk/corefs/%s_corefs.json' % resolver, 'r', encoding='utf-8') as f:
        cluster_lists = json.load(f)
    all_gt = []
    all_pred = []
    all_types_gt = [[],[],[],[],[]]
    all_types_pred = [[],[],[],[],[]]
    num_mentions = []
    for fusion_idx, cluster_list in enumerate(cluster_lists):
        pocs = fusions[fusion_idx]
        len_pocs = 0 if pocs[0][3] == '' else len(pocs)
        # poc_gt = np.ones((len_pocs), dtype=int)
        # poc_pred = np.zeros((len_pocs), dtype=int)
        # poc_types_gt = np.zeros((len(poc_unique_types), len_pocs, 3), dtype=int)
        # poc_types_pred = np.zeros((len(poc_unique_types), len_pocs, 3), dtype=int)
        # cluster_gt = []
        # cluster_pred = []
        # cluster_types_gt = [[],[],[],[],[]]
        # cluster_types_pred = [[],[],[],[],[]]

        cluster_list_only_pairs = [cluster for cluster in cluster_list if cluster_in_both_sents(cluster)]
        # cluster_list_only_pairs = [only_first_in_each_sent(cluster) for cluster in cluster_list_only_pairs]

        cluster_found_for_pocs = [0] * len_pocs
        cluster_found_for_pocs_types = [[], [], [], [], []]
        for poc_idx in range(len_pocs):
            poc = pocs[poc_idx]
            sent1_selection_indices = poc[4]
            sent2_selection_indices = poc[7]
            sent3_selection_indices = poc[10]
            poc_type = poc[11]
            poc_type_idx = poc_unique_types.index(poc_type)
            ranges = [sent1_selection_indices, sent2_selection_indices]

            found_mention = [False, False]
            for cluster in cluster_list_only_pairs:
                found_mention = [False, False]
                for mention in cluster:
                    mention_range = [mention['startIndex']-1, mention['endIndex']-1]
                    sent_idx = mention['sentNum']-1
                    gt_range = ranges[sent_idx]
                    if not (mention_range[1] <= gt_range[0] or mention_range[0] >= gt_range[1]):  # if the ranges overlap
                        found_mention[sent_idx] = True
                if found_mention[0] and found_mention[1]:
                    cluster_found_for_pocs[poc_idx] = 1
                    cluster_found_for_pocs_types[poc_type_idx].append(1)
                    break
            if not (found_mention[0] and found_mention[1]):
                cluster_found_for_pocs_types[poc_type_idx].append(0)

        num_hits = sum(cluster_found_for_pocs)
        num_misses = len(cluster_found_for_pocs) - num_hits
        num_bad_clusters = max(0, len(cluster_list_only_pairs) - num_hits)

        gt = ([1] * num_hits) + ([1] * num_misses) + ([0] * num_bad_clusters)
        pred = ([1] * num_hits) + ([0] * num_misses) + ([1] * num_bad_clusters)
        all_gt.extend(gt)
        all_pred.extend(pred)

        for poc_type_idx in range(5):
            num_hits = sum(cluster_found_for_pocs_types[poc_type_idx])
            num_misses = len(cluster_found_for_pocs_types[poc_type_idx]) - num_hits
            num_bad_clusters = max(0, len(cluster_list_only_pairs) - num_hits)
            gt = ([1] * num_hits) + ([1] * num_misses) + ([0] * num_bad_clusters)
            pred = ([1] * num_hits) + ([0] * num_misses) + ([1] * num_bad_clusters)
            all_types_gt[poc_type_idx].extend(gt)
            all_types_pred[poc_type_idx].extend(pred)


        # for cluster in cluster_list:
        #     num_mentions.append(len(cluster))
        #     for mention in cluster:
        #         mention_range = [mention['startIndex']-1, mention['endIndex']-1]
        #         gt_poc_found = False
        #         gt_poc_type_found = [False] * 5
        #         sent_idx = mention['sentNum']-1
        #         for poc_idx, poc in enumerate(pocs):
        #             sent1_selection_indices = poc[4]
        #             sent2_selection_indices = poc[7]
        #             sent3_selection_indices = poc[10]
        #             ranges = [sent1_selection_indices, sent2_selection_indices, sent3_selection_indices]
        #
        #             gt_range = ranges[sent_idx]
        #             if gt_range != []:
        #                 poc_type = poc[11]
        #                 poc_type_idx = poc_unique_types.index(poc_type)
        #                 poc_types_gt[poc_type_idx, poc_idx, sent_idx] = 1
        #                 if not (mention_range[1] <= gt_range[0] or mention_range[0] >= gt_range[1]):  # if the ranges overlap
        #                     gt_poc_found = True
        #                     poc_pred[poc_idx, sent_idx] = 1
        #                     gt_poc_type_found[poc_type_idx] = True
        #                     poc_types_pred[poc_type_idx, poc_idx, sent_idx] = 1
        #
        #         if not gt_poc_found:
        #             cluster_gt.append(0)
        #             cluster_pred.append(1)
        #         for poc_type_idx in range(5):
        #             if not gt_poc_type_found[poc_type_idx]:
        #                 cluster_types_gt[poc_type_idx].append(0)
        #                 cluster_types_pred[poc_type_idx].append(1)
        #
        #
        # cluster_gt.extend([1] * (len_pocs * 3))
        # cluster_pred.extend(flatten_list_of_lists(poc_pred.tolist()))
        # all_gt.extend(cluster_gt)
        # all_pred.extend(cluster_pred)
        #
        # for poc_type_idx in range(5):
        #     for i in range(poc_types_gt.shape[1]):
        #         for j in range(poc_types_gt.shape[2]):
        #             if poc_types_gt[poc_type_idx,i,j] == 1:
        #                 cluster_types_gt[poc_type_idx].append(1)
        #                 cluster_types_pred[poc_type_idx].append(poc_types_pred[poc_type_idx, i, j])
        #     all_types_gt[poc_type_idx].extend(cluster_types_gt[poc_type_idx])
        #     all_types_pred[poc_type_idx].extend(cluster_types_pred[poc_type_idx])

    # res = word_tag_eval(all_gt, all_pred)
    # scores = only_numbers(res)
    # print_scores(pretty_resolver, scores)

    # num_mentions_pmf = Pmf(num_mentions)
    # num_mentions_pmf.normalize()
    # print('Num mentions PMF: ', num_mentions_pmf)

    recalls = []
    for poc_type_idx, poc_type in enumerate(poc_unique_types):
        res = word_tag_eval(all_types_gt[poc_type_idx], all_types_pred[poc_type_idx])
        # print(poc_type)
        scores = only_numbers(res)
        recalls.append(scores[1])
    print_recalls(pretty_resolver, recalls)





print('done')






















