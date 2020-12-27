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

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""
import glob
import nltk
import os
import numpy as np

import tensorflow as tf
import util
from tqdm import tqdm
from absl import flags
from absl import logging
import rouge_functions
import json

import sys
import spacy
import bert_score
nlp = spacy.load('en_core_web_sm')

path_to_this_file = os.path.realpath(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(path_to_this_file),'bert'))
import run_decoding

FLAGS = flags.FLAGS


SECS_UNTIL_NEW_CKPT = 60 # max number of seconds before loading new checkpoint
threshold = 0.5
prob_to_keep = 0.33
pos_types = '''ADJ
ADP
ADV
AUX
CONJ
CCONJ
DET
INTJ
NOUN
NUM
PART
PRON
PROPN
PUNCT
SCONJ
SYM
VERB
X
SPACE'''.split('\n')


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher):
        """Initialize decoder.

        Args:
            model: a Seq2SeqAttentionModel object.
            batcher: a Batcher object.
        """
        self._batcher = batcher
        self._model = model
        if not FLAGS.unilm_decoding:
            self._model.build_graph()
            self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
            self._sess = tf.Session(config=util.get_config())

        if not FLAGS.unilm_decoding:
            # Load an initial checkpoint to use for decoding
            ckpt_path = util.load_ckpt(self._saver, self._sess)

            # Make a descriptive decode directory name
            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
            self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))

        else: # Generic decode dir name
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

        # Make the dirs to contain output written in the correct format for pyrouge
        self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
        if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
        self._human_dir = os.path.join(self._decode_dir, "human_readable")
        if not os.path.exists(self._human_dir): os.mkdir(self._human_dir)


    def append_corr_feature(self, corr_features, key, value):
        if key not in corr_features:
            corr_features[key] = []
        corr_features[key].append(value)


    def flatten_coref_chains(self, coref_chains, raw_article_sents, ssi):
        article_sent1 = raw_article_sents[ssi[0]]
        article_sent2 = raw_article_sents[ssi[1]]
        num_tokens_sent1 = len(article_sent1.split(' '))
        num_tokens_sent2 = len(article_sent2.split(' '))
        flat_coref_chains = []
        for chain in coref_chains:
            flat_chain = []
            for mention in chain:
                if mention[0] == 1:
                    flat_mention = (num_tokens_sent1 + mention[1], num_tokens_sent1 + mention[2])
                else:
                    flat_mention = (mention[1], mention[2])
                flat_chain.append(flat_mention)
            flat_coref_chains.append(flat_chain)
        return flat_coref_chains

    def decode_iteratively(self, source_data_path, total, ssi_list):
        if FLAGS.dataset_name == 'xsum':
            l_param = 100
        else:
            l_param = 100
        all_gt_word_tags = []
        all_sys_word_tags = []
        sources_lens = []
        perc_words_tagged = []
        if FLAGS.unilm_decoding:
            bert_run = run_decoding.BertRun()
            FLAGS.do_predict = True
            bert_run.setUpModel()
            bert_run.setUpPredict()
        with open(source_data_path) as f:
            f_text = f.read()
        lines = f_text.strip().split('\n')
        data_lines = lines[1:]
        for example_idx, example_data_line in enumerate(tqdm(data_lines)):
            article_sents, groundtruth_summary_text, _, _, sent_2_start_token_idx, coref_chains_str = example_data_line.strip().split('\t')
            sentence_ids = [0, 1]
            sent2_start = int(sent_2_start_token_idx)
            article_tokens = article_sents.strip().split()
            raw_article_sents = [' '.join(article_tokens[:sent2_start]), ' '.join(article_tokens[sent2_start:])]
            groundtruth_article_lcs_paths_list = [[list(range(len(sent))) for sent in raw_article_sents]]
            groundtruth_similar_source_indices_list = [[0, 1]]
            if FLAGS.link:
                coref_chains_dict = json.loads(coref_chains_str)
                coref_chains = []
                for chain_id in sorted(list(coref_chains_dict.keys())):
                    chain_to_add = []
                    for mention in coref_chains_dict[chain_id]:
                        chain_to_add.append((mention['start'], mention['end']))
                    coref_chains.append(chain_to_add)
            else:
                coref_chains = None
            if FLAGS.first_chain_only:
                coref_chains = [coref_chains[0]]

            article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
            groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
            groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]

            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break

            if ssi_list is None:    # this is if we are doing the upper bound evaluation (ssi_list comes straight from the groundtruth)
                sys_ssi = groundtruth_similar_source_indices_list
                sys_alp_list = groundtruth_article_lcs_paths_list
                sys_ssi = util.enforce_sentence_limit(sys_ssi, 2)
                sys_alp_list = util.enforce_sentence_limit(sys_alp_list, 2)
                sys_ssi, sys_alp_list = util.replace_empty_ssis(sys_ssi, raw_article_sents, sys_alp_list=sys_alp_list)
            else:
                gt_ssi, sys_ssi, ext_len = ssi_list[example_idx]
                sys_alp_list = [[list(range(len(article_sent_tokens[source_idx]))) for source_idx in source_indices] for source_indices in sys_ssi]
                sys_alp_list_for_gt_ssi = groundtruth_article_lcs_paths_list

                sys_ssi = util.enforce_sentence_limit(sys_ssi, 2)
                sys_alp_list = util.enforce_sentence_limit(sys_alp_list, 2)
                sys_alp_list_for_gt_ssi = util.enforce_sentence_limit(sys_alp_list_for_gt_ssi, 2)
                groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, 2)
                gt_ssi = util.enforce_sentence_limit(gt_ssi, 2)

            final_decoded_words = []
            final_decoded_sent_tokens = []
            final_decoded_outpus = ''
            best_hyps = []
            my_sources = []
            for ssi_idx, ssi in enumerate(sys_ssi):

                selected_article_lcs_paths = sys_alp_list[ssi_idx]
                ssi, selected_article_lcs_paths = util.make_ssi_chronological(ssi, selected_article_lcs_paths)
                selected_article_lcs_paths = [selected_article_lcs_paths]

                selected_raw_article_sents = util.reorder(raw_article_sents, ssi)
                selected_article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in selected_raw_article_sents]
                selected_article_text = ' '.join( [' '.join(sent) for sent in util.reorder(article_sent_tokens, ssi)] )
                selected_doc_indices_str = '0 ' * len(selected_article_text.split())
                selected_groundtruth_summ_sent = groundtruth_summ_sents
                my_sources.extend(util.flatten_list_of_lists(selected_article_sent_tokens))

                if FLAGS.unilm_decoding:
                    text_a = ' '.join(selected_raw_article_sents)
                    sent2_start = len(selected_raw_article_sents[0])
                    text_b = selected_groundtruth_summ_sent[0][0]
                    decoded_words = bert_run.predict(text_a, text_b, coref_chains, sent2_start)
                    decoded_output = ' '.join(decoded_words)
                final_decoded_words.extend(decoded_words)
                final_decoded_sent_tokens.append(decoded_words)
                final_decoded_outpus += decoded_output
                final_decoded_sent = [' '.join(decoded_words)]


                if len(final_decoded_words) >= 100:
                    break

            sources_lens.append(len(my_sources))
            gt_ssi_list, gt_alp_list = util.replace_empty_ssis(groundtruth_similar_source_indices_list, raw_article_sents, sys_alp_list=groundtruth_article_lcs_paths_list)
            for ssi_idx, ssi in enumerate(gt_ssi_list):
                selected_article_lcs_paths = gt_alp_list[ssi_idx]
                try:
                    ssi, selected_article_lcs_paths = util.make_ssi_chronological(ssi, selected_article_lcs_paths)
                except:
                    util.print_vars(ssi, example_idx, selected_article_lcs_paths)
                    raise
                selected_raw_article_sents = util.reorder(raw_article_sents, ssi)

            self.write_for_human(raw_article_sents, groundtruth_summ_sents, final_decoded_sent_tokens, example_idx, already_sent_split=True)


            final_decoded_sents = [' '.join(sent) for sent in final_decoded_sent_tokens]
            rouge_functions.write_for_rouge(groundtruth_summ_sents, final_decoded_sents, example_idx, self._rouge_ref_dir, self._rouge_dec_dir, log=False) # write ref summary and decoded summary to file, to eval with pyrouge later
            example_idx += 1 # this is how many examples we've decoded

        if len(os.listdir(self._rouge_ref_dir)) != 0:
            logging.info("Now starting ROUGE eval...")
            results_dict = rouge_functions.rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir, l_param=l_param)

            sheets_results_text = rouge_functions.rouge_log(results_dict, self._decode_dir)

            decoded_files = sorted(glob.glob(os.path.join(self._rouge_dec_dir, '*')))
            reference_files = sorted(glob.glob(os.path.join(self._rouge_ref_dir, '*')))
            cands = []
            refs = []
            lens = []
            for file in decoded_files:
                with open(file) as f:
                    text = f.read().replace('\n', ' ')
                    cands.append(text)
                    lens.append(len(text.strip().split()))
            for file in reference_files:
                with open(file) as f:
                    refs.append(f.read().replace('\n', ' '))
            bleu = nltk.translate.bleu_score.corpus_bleu([[ref] for ref in refs], cands) * 100
            bert_p, bert_r, bert_f = bert_score.score(cands, refs, lang="en", verbose=False, batch_size=8,
                                                      model_type='bert-base-uncased')
            avg_len = np.mean(lens)
            bert_p = np.mean(bert_p.cpu().numpy()) * 100
            bert_r = np.mean(bert_r.cpu().numpy()) * 100
            bert_f = np.mean(bert_f.cpu().numpy()) * 100
            sheets_results_contents = sheets_results_text.strip().split('\t')
            new_results = ['%.2f' % avg_len] + sheets_results_contents + ['%.2f' % bert_p, '%.2f' % bert_r,
                                                                          '%.2f' % bert_f, '%.2f' % bleu]
            print('\t'.join(new_results))
            new_sheets_results_file = os.path.join(self._decode_dir, 'bert_sheets_results.txt')
            with open(new_sheets_results_file, 'w') as f:
                f.write('\t'.join(new_results))



    def write_for_human(self, raw_article_sents, all_reference_sents, decoded_words, ex_index, already_sent_split=False):

        if already_sent_split:
            decoded_sents = [' '.join(sent_tokens) for sent_tokens in decoded_words]
        else:
            decoded_sents = []
            while len(decoded_words) > 0:
                try:
                    fst_period_idx = decoded_words.index(".")
                except ValueError: # there is text remaining that doesn't end in "."
                    fst_period_idx = len(decoded_words)
                sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
                decoded_words = decoded_words[fst_period_idx+1:] # everything else
                decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        all_reference_sents = [[make_html_safe(w) for w in abstract] for abstract in all_reference_sents]

        # Write to file
        human_file = os.path.join(self._human_dir, '%06d_human.txt' % ex_index)

        with open(human_file, "w") as f:
            f.write('Human Summary:\n--------------------------------------------------------------\n')
            for abs_idx, abs in enumerate(all_reference_sents):
                for idx,sent in enumerate(abs):
                    f.write(sent+"\n")
            f.write('\nSystem Summary:\n--------------------------------------------------------------\n')
            for sent in decoded_sents:
                f.write(sent + '\n')
            f.write('\nArticle:\n--------------------------------------------------------------\n')
            for sent in raw_article_sents:
                f.write(sent + '\n')


def print_results(article, abstract, decoded_output):
    """Prints the article, the reference summmary and the decoded summary to screen"""
    print("")
    logging.info('ARTICLE:	%s', article)
    logging.info('REFERENCE SUMMARY: %s', abstract)
    logging.info('GENERATED SUMMARY: %s', decoded_output)
    print("")


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def get_decode_dir_name(ckpt_name):
    """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

    if "train" in FLAGS.data_path: dataset = "train"
    elif "val" in FLAGS.data_path: dataset = "val"
    elif "test" in FLAGS.data_path: dataset = "test"
    else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
    # dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    dirname = "decode"
    dirname += "_%imaxenc_%imindec_%imaxdec" % (FLAGS.max_enc_steps, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
