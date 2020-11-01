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
import time
import numpy as np

import pickle
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
from sumy.nlp.tokenizers import Tokenizer
from tqdm import tqdm
from absl import flags
from absl import logging
import logging as log
import rouge_functions

import importance_features
from batcher import create_batch
import ssi_functions
import sys
from scipy.stats.stats import pearsonr
import spacy
from collections import Counter
import bert_score
# import neuralcoref
nlp = spacy.load('en_core_web_sm')
# neuralcoref.add_to_pipe(nlp)

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

    def __init__(self, model, batcher, vocab):
        """Initialize decoder.

        Args:
            model: a Seq2SeqAttentionModel object.
            batcher: a Batcher object.
            vocab: Vocabulary object
        """
        self._batcher = batcher
        self._vocab = vocab
        self._model = model
        if not FLAGS.unilm_decoding:
            self._model.build_graph()
            self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
            self._sess = tf.Session(config=util.get_config())

        if not FLAGS.unilm_decoding:
            # Load an initial checkpoint to use for decoding
            ckpt_path = util.load_ckpt(self._saver, self._sess)

            if FLAGS.single_pass:
                # Make a descriptive decode directory name
                ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
                self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
    # 			if os.path.exists(self._decode_dir):
    # 				raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        else: # Generic decode dir name
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

        if FLAGS.single_pass:
            # Make the dirs to contain output written in the correct format for pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
            self._human_dir = os.path.join(self._decode_dir, "human_readable")
            if not os.path.exists(self._human_dir): os.mkdir(self._human_dir)
            self._highlight_dir = os.path.join(self._decode_dir, "highlight")
            if not os.path.exists(self._highlight_dir): os.mkdir(self._highlight_dir)
            self._tmp_dir = os.path.join(self._decode_dir, "tmp")
            if not os.path.exists(self._tmp_dir): os.mkdir(self._tmp_dir)
            self._other_tmp_dir = os.path.join(self._decode_dir, "tmp_other")
            if not os.path.exists(self._other_tmp_dir): os.mkdir(self._other_tmp_dir)
            self._better_dir = os.path.join(self._decode_dir, "better")
            if not os.path.exists(self._better_dir): os.mkdir(self._better_dir)
            self._worse_dir = os.path.join(self._decode_dir, "worse")
            if not os.path.exists(self._worse_dir): os.mkdir(self._worse_dir)

            if FLAGS.worker_idx != -1:
                self._worker_signal_dir = os.path.join(self._decode_dir, "worker_signals")
                if not os.path.exists(self._worker_signal_dir): os.mkdir(self._worker_signal_dir)
                self.write_worker_signal(False)


    def write_worker_signal(self, is_done):
        with open(os.path.join(self._worker_signal_dir, '%d_signal.txt' % FLAGS.worker_idx), 'wb') as f:
            f.write(str(is_done).encode())

    def check_workers_done(self):
        for worker_idx in range(FLAGS.num_workers):
            # while True:
            #     try:
            #         with open(os.path.join(self._worker_signal_dir, '%d_signal.txt' % worker_idx)) as f:
            #             if f.read().strip() == 'False':
            #                 return False
            #     except:
            #         continue
            with open(os.path.join(self._worker_signal_dir, '%d_signal.txt' % worker_idx)) as f:
                if f.read().strip() == 'False':
                    return False
        return True

    def append_corr_feature(self, corr_features, key, value):
        if key not in corr_features:
            corr_features[key] = []
        corr_features[key].append(value)


    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        t0 = time.time()
        counter = 0
        attn_dir = os.path.join(self._decode_dir, 'attn_vis_data')
        total = len(glob.glob(self._batcher._data_path)) * 1000
        pbar = tqdm(total=total)
        while True:
            batch = self._batcher.next_batch()	# 1 example repeated across batch
            if batch is None: # finished decoding dataset in single_pass mode
                assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                # attn_selections.process_attn_selections(attn_dir, self._decode_dir, self._vocab)
                logging.info("Decoder has finished reading dataset for single_pass.")
                logging.info("Output has been saved in %s and %s.", self._rouge_ref_dir, self._rouge_dec_dir)
                if len(os.listdir(self._rouge_ref_dir)) != 0:
                    logging.info("Now starting ROUGE eval...")
                    results_dict = rouge_functions.rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                    rouge_functions.rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]   # string
            original_abstract = batch.original_abstracts[0]		# string
            all_original_abstract_sents = batch.all_original_abstracts_sents[0]
            raw_article_sents = batch.raw_article_sents[0]

            article_withunks = data.show_art_oovs(original_article, self._vocab) # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

            decoded_words, decoded_output, best_hyp = decode_example(self._sess, self._model, self._vocab, batch, counter, self._batcher._hps)

            if FLAGS.single_pass:
                if counter < 1000:
                    self.write_for_human(raw_article_sents, all_original_abstract_sents, decoded_words, counter)
                rouge_functions.write_for_rouge(all_original_abstract_sents, None, counter, self._rouge_ref_dir, self._rouge_dec_dir, decoded_words=decoded_words) # write ref summary and decoded summary to file, to eval with pyrouge later
                if FLAGS.attn_vis:
                    self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens, counter) # write info to .json file for visualization tool

                    # if counter % 1000 == 0:
                    #     attn_selections.process_attn_selections(attn_dir, self._decode_dir, self._vocab)

                counter += 1 # this is how many examples we've decoded
            else:
                print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens, counter) # write info to .json file for visualization tool

                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
                t1 = time.time()
                if t1-t0 > SECS_UNTIL_NEW_CKPT:
                    logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()
            pbar.update(1)
        pbar.close()

    def flatten_coref_chains(self, coref_chains, raw_article_sents, ssi):
        article_sent1 = raw_article_sents[ssi[0]]
        article_sent2 = raw_article_sents[ssi[1]]
        num_tokens_sent1 = len(article_sent1.split(' '))
        num_tokens_sent2 = len(article_sent2.split(' '))
        flat_coref_chains = []
        for chain in coref_chains:
            flat_chain = []
            for mention in chain:
                if FLAGS.triples_only:
                    if mention[0] == 2:
                        flat_mention = (num_tokens_sent1 + mention[1], num_tokens_sent1 + mention[2])
                    elif mention[0] == 0:
                        flat_mention = (num_tokens_sent1 + num_tokens_sent2 + mention[1],
                                        num_tokens_sent1 + num_tokens_sent2 + mention[2])
                    else:
                        flat_mention = (mention[1], mention[2])
                else:
                    if mention[0] == 1:
                        flat_mention = (num_tokens_sent1 + mention[1], num_tokens_sent1 + mention[2])
                    else:
                        flat_mention = (mention[1], mention[2])
                flat_chain.append(flat_mention)
            flat_coref_chains.append(flat_chain)
        return flat_coref_chains

    def decode_iteratively(self, example_generator, total, names_to_types, ssi_list, hps):
        if FLAGS.dataset_name == 'xsum':
            l_param = 100
        else:
            l_param = 100
        if not FLAGS.only_rouge:
            attn_vis_idx = 0
            all_gt_word_tags = []
            all_sys_word_tags = []
            sources_lens = []
            perc_words_tagged = []
            better_idx = 0
            worse_idx = 0
            corr_features = {}
            corr_outputs = {}
            if FLAGS.worker_idx != -1:
                start_ex_idx = FLAGS.worker_idx * (total // FLAGS.num_workers)
                if FLAGS.worker_idx == FLAGS.num_workers-1:     # if this is the last worker, then he takes the leftover all the way to the end
                    end_ex_idx = np.inf
                else:
                    end_ex_idx = (FLAGS.worker_idx + 1) * (total // FLAGS.num_workers)
            if FLAGS.unilm_decoding:
                bert_run = run_decoding.BertRun()
                FLAGS.do_predict = True
                bert_run.setUpModel()
                bert_run.setUpPredict()
            for example_idx, example in enumerate(tqdm(example_generator, total=total)):
                if FLAGS.worker_idx != -1 and (example_idx < start_ex_idx or example_idx >= end_ex_idx):     # This worker will ignore examples that are outside his domain
                    continue

                if FLAGS.poc_dataset:
                    raw_article_sents, groundtruth_summary_text, original_coref_chains = util.unpack_tf_example(example, names_to_types)
                    if FLAGS.poc_gold:
                        coref_chains = self.flatten_coref_chains(original_coref_chains, raw_article_sents, [0,1])
                    else:
                        coref_chains, article_sent_tokens = get_coref_chains(sent1, sent2)
                    corefs = None
                    groundtruth_similar_source_indices_list = [[0,1]]
                    groundtruth_article_lcs_paths_list = [[list(range(len(sent))) for sent in raw_article_sents]]
                elif FLAGS.triples_only:
                    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, groundtruth_article_lcs_paths_list, original_coref_chains = util.unpack_tf_example(
                        example, names_to_types)
                    groundtruth_similar_source_indices_list = [[0,1]]
                    groundtruth_article_lcs_paths_list = [[list(range(len(sent))) for sent in raw_article_sents]]
                    coref_chains = self.flatten_coref_chains(original_coref_chains, raw_article_sents,
                                                             groundtruth_similar_source_indices_list[0])
                    corefs = None
                elif FLAGS.coref_dataset:
                    # raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, groundtruth_article_lcs_paths_list, coref_chains, coref_representatives = util.unpack_tf_example(
                    #     example, names_to_types)
                    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, groundtruth_article_lcs_paths_list, original_coref_chains = util.unpack_tf_example(
                        example, names_to_types)
                    coref_chains = self.flatten_coref_chains(original_coref_chains, raw_article_sents, groundtruth_similar_source_indices_list[0])
                    corefs = None
                else:
                    raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, groundtruth_article_lcs_paths_list = util.unpack_tf_example(
                        example, names_to_types)
                    coref_chains = None
                    original_coref_chains = None
                article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
                groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
                groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]

                if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                    break

                if ssi_list is None:    # this is if we are doing the upper bound evaluation (ssi_list comes straight from the groundtruth)
                    sys_ssi = groundtruth_similar_source_indices_list
                    sys_alp_list = groundtruth_article_lcs_paths_list
                    if FLAGS.singles_and_pairs == 'singles':
                        sys_ssi = util.enforce_sentence_limit(sys_ssi, 1)
                        sys_alp_list = util.enforce_sentence_limit(sys_alp_list, 1)
                    elif FLAGS.singles_and_pairs == 'both':
                        sys_ssi = util.enforce_sentence_limit(sys_ssi, 2)
                        sys_alp_list = util.enforce_sentence_limit(sys_alp_list, 2)
                    sys_ssi, sys_alp_list = util.replace_empty_ssis(sys_ssi, raw_article_sents, sys_alp_list=sys_alp_list)
                else:
                    if FLAGS.tag_tokens:
                        gt_ssi, sys_ssi, ext_len, sys_token_probs_list, gt_word_tags, sys_token_probs_list_for_gt_ssi = ssi_list[example_idx]
                        if FLAGS.ninst != -1:
                            sys_ssi = sys_ssi[:FLAGS.ninst]
                            sys_token_probs_list = sys_token_probs_list[:FLAGS.ninst]
                        binarize_parameter = FLAGS.tag_threshold if FLAGS.binarize_method == 'threshold' else FLAGS.summ_limit if FLAGS.binarize_method == 'summ_limit' else FLAGS.inst_limit if FLAGS.binarize_method == 'inst_limit' else None
                        sys_alp_list = ssi_functions.list_labels_from_probs(sys_token_probs_list, FLAGS.binarize_method, binarize_parameter)
                        sys_alp_list_for_gt_ssi = ssi_functions.list_labels_from_probs(sys_token_probs_list_for_gt_ssi, FLAGS.binarize_method, binarize_parameter)
                    else:
                            gt_ssi, sys_ssi, ext_len = ssi_list[example_idx]
                            sys_alp_list = [[list(range(len(article_sent_tokens[source_idx]))) for source_idx in source_indices] for source_indices in sys_ssi]
                            sys_alp_list_for_gt_ssi = groundtruth_article_lcs_paths_list
                    # if FLAGS.binarize_method == 'summ_limit':
                    #     sys_ssi = sys_ssi[:ext_len]
                    #     sys_token_probs_list = sys_token_probs_list[:ext_len]
                    if example_idx == 239:
                        a=0
                    if FLAGS.upper_bound_ssi_only:
                        sys_ssi = gt_ssi
                        sys_alp_list = sys_alp_list_for_gt_ssi
                        sys_ssi, sys_alp_list = util.replace_empty_ssis(sys_ssi, raw_article_sents, sys_alp_list=sys_alp_list)
                    if FLAGS.singles_and_pairs == 'singles':
                        sys_ssi = util.enforce_sentence_limit(sys_ssi, 1)
                        sys_alp_list = util.enforce_sentence_limit(sys_alp_list, 1)
                        sys_alp_list_for_gt_ssi = util.enforce_sentence_limit(sys_alp_list_for_gt_ssi, 1)
                        groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, 1)
                        gt_ssi = util.enforce_sentence_limit(gt_ssi, 1)
                    elif FLAGS.singles_and_pairs == 'both':
                        sys_ssi = util.enforce_sentence_limit(sys_ssi, 2)
                        sys_alp_list = util.enforce_sentence_limit(sys_alp_list, 2)
                        sys_alp_list_for_gt_ssi = util.enforce_sentence_limit(sys_alp_list_for_gt_ssi, 2)
                        groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, 2)
                        gt_ssi = util.enforce_sentence_limit(gt_ssi, 2)

                    # if gt_ssi != groundtruth_similar_source_indices_list:
                    #     raise Exception('Example %d has different groundtruth source indices: ' + str(groundtruth_similar_source_indices_list) + ' || ' + str(gt_ssi))
                    if FLAGS.dataset_name == 'xsum':
                        sys_ssi = [sys_ssi[0]]


                if FLAGS.word_eval_only:
                    gt_tags = util.alp_list_to_binary_tags(groundtruth_article_lcs_paths_list, article_sent_tokens, groundtruth_similar_source_indices_list)
                    sys_tags = util.alp_list_to_binary_tags(sys_alp_list_for_gt_ssi, article_sent_tokens, gt_ssi)
                    all_gt_word_tags.extend(gt_tags)
                    all_sys_word_tags.extend(sys_tags)

                    sys_tagged = len(util.flatten_list_of_lists(util.flatten_list_of_lists(sys_alp_list)))
                    all_my_article_sent_tokens = 0
                    for source_indices in sys_ssi:
                        for source_idx in source_indices:
                            all_my_article_sent_tokens += len(article_sent_tokens[source_idx])
                    perc_words_tagged.append(sys_tagged * 1.0 / all_my_article_sent_tokens)
                else:

                    final_decoded_words = []
                    final_decoded_sent_tokens = []
                    final_decoded_outpus = ''
                    best_hyps = []
                    highlight_html_total = '<u>System Summary</u><br><br>'
                    highlight_html_tagged = '<br><br><u>Only tagged</u><br><br>'
                    highlight_html_other = '<u>Other System Summary (%s)</u><br><br>' % FLAGS.compare_exp_name
                    my_sources = []
                    for ssi_idx, ssi in enumerate(sys_ssi):
                        # selected_article_lcs_paths = None

                        # if FLAGS.tag_tokens:
                        #     selected_article_lcs_paths = sys_alp_list[ssi_idx]
                        #     ssi, selected_article_lcs_paths = util.make_ssi_chronological(ssi, selected_article_lcs_paths)
                        #     selected_article_lcs_paths = [selected_article_lcs_paths]
                        # else:
                        #     ssi = util.make_ssi_chronological(ssi)
                        #     selected_article_lcs_paths = [[]]

                        selected_article_lcs_paths = sys_alp_list[ssi_idx]
                        ssi, selected_article_lcs_paths = util.make_ssi_chronological(ssi, selected_article_lcs_paths)
                        selected_article_lcs_paths = [selected_article_lcs_paths]

                        selected_raw_article_sents = util.reorder(raw_article_sents, ssi)
                        selected_article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in selected_raw_article_sents]
                        selected_article_text = ' '.join( [' '.join(sent) for sent in util.reorder(article_sent_tokens, ssi)] )
                        selected_doc_indices_str = '0 ' * len(selected_article_text.split())
                        if FLAGS.upper_bound:
                            selected_groundtruth_summ_sent = [[groundtruth_summ_sents[0][ssi_idx]]]
                        else:
                            selected_groundtruth_summ_sent = groundtruth_summ_sents
                        my_sources.extend(util.flatten_list_of_lists(selected_article_sent_tokens))

                        batch = create_batch(selected_article_text, selected_groundtruth_summ_sent, selected_doc_indices_str, selected_raw_article_sents, selected_article_lcs_paths, FLAGS.batch_size, hps, self._vocab)

                        original_article = batch.original_articles[0]  # string
                        original_abstract = batch.original_abstracts[0]  # string
                        article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
                        abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                               (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string
                        # article_withunks = data.show_art_oovs(original_article, self._vocab) # string
                        # abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

                        pull_from_decode_dir_failed = False
                        if FLAGS.pull_from_decode_dir:
                            decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % example_idx)
                            with open(os.path.join(decoded_file)) as f:
                                decoded_output = f.read().strip()
                            decoded_words = decoded_output.strip().split(' ')
                            if len(decoded_words) == 0 or (len(decoded_words) == 1 and decoded_words[0] == ''):
                                pull_from_decode_dir_failed = True
                        elif FLAGS.first_intact and ssi_idx == 0:
                            decoded_words = selected_article_text.strip().split()
                            decoded_output = selected_article_text
                        elif FLAGS.only_tags:
                            fixed_selected_article_lcs_paths = []
                            for sent_idx, sel_alp in enumerate(selected_article_lcs_paths[0]):
                                if len(sel_alp) == 0:
                                    fixed_selected_article_lcs_paths.append([])
                                    continue
                                if max(sel_alp) >= len(selected_article_sent_tokens[sent_idx]):
                                    print('Warning: max(sel_alp) >= len(selected_article_sent_tokens[sent_idx])', max(sel_alp), len(selected_article_sent_tokens[sent_idx]))
                                fixed_selected_article_lcs_paths.append([word_idx for word_idx in sel_alp if word_idx < len(selected_article_sent_tokens[sent_idx])])
                            decoded_words = util.flatten_list_of_lists([util.reorder(selected_article_sent_tokens[sent_idx], sel_alp) for sent_idx, sel_alp in enumerate(fixed_selected_article_lcs_paths)] + ['.'])
                            decoded_output = ' '.join(decoded_words)
                        else:
                            if FLAGS.unilm_decoding:
                                if FLAGS.repl:
                                    raw_article_sent_tokens = util.coref_replace(raw_article_sents, ssi, coref_chains, coref_representatives)
                                    text_a = ' '.join([' '.join(sent) for sent in raw_article_sent_tokens])
                                    sent2_start = len(raw_article_sent_tokens[0])
                                else:
                                    text_a = ' '.join(selected_raw_article_sents)
                                    sent2_start = len(selected_raw_article_sents[0])
                                text_b = selected_groundtruth_summ_sent[0][0]
                                decoded_words = bert_run.predict(text_a, text_b, coref_chains, sent2_start)
                                decoded_output = ' '.join(decoded_words)
                            elif FLAGS.concatbaseline:
                                decoded_words = util.flatten_list_of_lists(selected_article_sent_tokens)
                                decoded_output = ' '.join(decoded_words)
                            else:
                                decoded_words, decoded_output, best_hyp = decode_example(self._sess, self._model, self._vocab, batch, example_idx, hps)
                                best_hyps.append(best_hyp)
                        if pull_from_decode_dir_failed:        # Copied chunk of code from directly above. We want to re-run the decoder if for some reason, the text in the xxxxxx_decoded.txt file was not saved properly.
                            if FLAGS.unilm_decoding:
                                if FLAGS.repl:
                                    raw_article_sent_tokens = util.coref_replace(raw_article_sents, ssi, coref_chains, coref_representatives)
                                    text_a = ' '.join([' '.join(sent) for sent in raw_article_sent_tokens])
                                    sent2_start = len(raw_article_sent_tokens[0])
                                else:
                                    text_a = ' '.join(selected_raw_article_sents)
                                    sent2_start = len(selected_raw_article_sents[0])
                                text_b = selected_groundtruth_summ_sent[0][0]
                                decoded_words = bert_run.predict(text_a, text_b, coref_chains, sent2_start)
                                decoded_output = ' '.join(decoded_words)
                            elif FLAGS.concatbaseline:
                                decoded_words = util.flatten_list_of_lists(selected_article_sent_tokens)
                                decoded_output = ' '.join(decoded_words)
                            else:
                                decoded_words, decoded_output, best_hyp = decode_example(self._sess, self._model, self._vocab, batch, example_idx, hps)
                                best_hyps.append(best_hyp)
                        final_decoded_words.extend(decoded_words)
                        final_decoded_sent_tokens.append(decoded_words)
                        final_decoded_outpus += decoded_output
                        final_decoded_sent = [' '.join(decoded_words)]

                        if FLAGS.compare_iter_rouge or FLAGS.corr_tests:
                            # util.delete_contents(self._tmp_dir)
                            # rouge_functions.write_for_rouge(groundtruth_summ_sents, final_decoded_sent, example_idx,
                            #                                 self._rouge_ref_dir, self._tmp_dir,
                            #                                 log=False)  # write ref summary and decoded summary to file, to eval with pyrouge later
                            # my_results_dict = rouge_functions.rouge_eval(self._rouge_ref_dir, self._tmp_dir, l_param=l_param)
                            #
                            other_decoded_file = os.path.join('logs', FLAGS.compare_exp_name, 'decode', 'decoded', "%06d_decoded.txt" % example_idx)
                            with open(os.path.join(other_decoded_file)) as f:
                                other_decoded_output = f.read().strip()
                            other_decoded_words = other_decoded_output.split(' ')
                            # other_final_decoded_sent = [' '.join(other_decoded_words)]
                            # util.delete_contents(self._other_tmp_dir)
                            # rouge_functions.write_for_rouge(groundtruth_summ_sents, other_final_decoded_sent, example_idx,
                            #                                 self._rouge_ref_dir, self._other_tmp_dir,
                            #                                 log=False)  # write ref summary and decoded summary to file, to eval with pyrouge later
                            # other_results_dict = rouge_functions.rouge_eval(self._rouge_ref_dir, self._other_tmp_dir, l_param=l_param)
                            #
                            # my_r1, my_r2, my_rl = my_results_dict['rouge_1_f_score'], my_results_dict['rouge_2_f_score'], my_results_dict['rouge_l_f_score']
                            # o_r1, o_r2, o_rl = other_results_dict['rouge_1_f_score'], other_results_dict['rouge_2_f_score'], other_results_dict['rouge_l_f_score']

                            my_r1 = rouge_functions.rouge_1(final_decoded_words, groundtruth_summ_sent_tokens[0], 0.5)
                            my_r2 = rouge_functions.rouge_2(final_decoded_words, groundtruth_summ_sent_tokens[0], 0.5)
                            my_rl = util.calc_ROUGE_L_score(final_decoded_words, groundtruth_summ_sent_tokens[0])

                            o_r1 = rouge_functions.rouge_1(other_decoded_words, groundtruth_summ_sent_tokens[0], 0.5)
                            o_r2 = rouge_functions.rouge_2(other_decoded_words, groundtruth_summ_sent_tokens[0], 0.5)
                            o_rl = util.calc_ROUGE_L_score(other_decoded_words, groundtruth_summ_sent_tokens[0])

                            # print(my_r1)
                            # print(my_r2)
                            # print(my_rl)
                            # print(o_r1)
                            # print(o_r2)
                            # print(o_rl)
                            # print('----------------------')

                            if FLAGS.compare_iter_rouge:
                                is_better = util.rouge_significantly_better(my_r1, my_r2, my_rl, o_r1, o_r2, o_rl)
                                is_worse = util.rouge_significantly_worse(my_r1, my_r2, my_rl, o_r1, o_r2, o_rl)
                            else:
                                is_better = False
                                is_worse = False

                            if FLAGS.corr_tests:
                                doc = nlp(' '.join(selected_raw_article_sents))
                                self.append_corr_feature(corr_features, 'length', len(util.flatten_list_of_lists(selected_article_sent_tokens)))
                                self.append_corr_feature(corr_features, 'num_coref_chains', len(coref_chains))
                                self.append_corr_feature(corr_features, 'length_diff', abs(len(selected_article_sent_tokens[0]) - len(selected_article_sent_tokens[1])))
                                num_entities = len([X for X in doc.ents])
                                self.append_corr_feature(corr_features, 'num_entities', num_entities)
                                self.append_corr_feature(corr_features, 'summ_length', len(groundtruth_summ_sent_tokens[0]))
                                # coref_clusters = doc._.coref_clusters
                                # self.append_corr_feature(corr_features, 'num_coref_clusters', len(coref_clusters))
                                # coref_mentions = len(util.flatten_list_of_lists(coref_clusters))
                                # self.append_corr_feature(corr_features, 'num_coref_mentions', len(coref_mentions))
                                pos_counter = Counter()
                                for token in doc:
                                    pos_counter[token.pos_] += 1
                                for pos in pos_types:
                                    if pos in pos_counter:
                                        count = pos_counter[pos]
                                    else:
                                        count = 0
                                    self.append_corr_feature(corr_features, pos, count)

                                my_ave_r = (my_r1 + my_r2 + my_rl)/3.
                                o_ave_r = (o_r1 + o_r2 + o_rl)/3.
                                self.append_corr_feature(corr_outputs, 'my_rouge', my_ave_r)
                                self.append_corr_feature(corr_outputs, 'rouge_diff', my_ave_r - o_ave_r)
                        else:
                            is_better = False
                            is_worse = False

                        if example_idx < 100 or (example_idx >= 2000 and example_idx < 2100) or is_better or is_worse:
                            min_matched_tokens = 2
                            selected_article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in selected_raw_article_sents]
                            highlight_summary_sent_tokens = [decoded_words]
                            highlight_ssi_list, lcs_paths_list, highlight_article_lcs_paths_list, highlight_smooth_article_lcs_paths_list = ssi_functions.get_simple_source_indices_list(
                                highlight_summary_sent_tokens,
                                selected_article_sent_tokens, 2, min_matched_tokens)
                            highlighted_html = ssi_functions.html_highlight_sents_in_article(highlight_summary_sent_tokens,
                                                                                           highlight_ssi_list,
                                                                                             selected_article_sent_tokens,
                                                                                           lcs_paths_list=lcs_paths_list,
                                                                                           article_lcs_paths_list=highlight_smooth_article_lcs_paths_list,
                                                                                             fusion_locations=original_coref_chains
                                                                                             )
                                                                                           # gt_similar_source_indices_list=[list(range(len(ssi)))],
                                                                                           # gt_article_lcs_paths_list=selected_article_lcs_paths)
                            highlight_html_total += highlighted_html + '<br>'

                            if FLAGS.tag_tokens:
                                my_highlight_html_tagged = ssi_functions.html_highlight_sents_in_article(highlight_summary_sent_tokens,
                                                                                                  [list(range(len(ssi)))],
                                                                                                 selected_article_sent_tokens,
                                                                                               lcs_paths_list=None,
                                                                                               article_lcs_paths_list=selected_article_lcs_paths,
                                                                                                 fusion_locations=original_coref_chains
                                                                                                         )
                                highlight_html_tagged += my_highlight_html_tagged + '<br>'

                        if is_better or is_worse:
                            min_matched_tokens = 2
                            selected_article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in selected_raw_article_sents]
                            highlight_summary_sent_tokens = [other_decoded_words]
                            highlight_ssi_list, lcs_paths_list, highlight_article_lcs_paths_list, highlight_smooth_article_lcs_paths_list = ssi_functions.get_simple_source_indices_list(
                                highlight_summary_sent_tokens,
                                selected_article_sent_tokens, 2, min_matched_tokens)
                            highlighted_html = ssi_functions.html_highlight_sents_in_article(highlight_summary_sent_tokens,
                                                                                           highlight_ssi_list,
                                                                                             selected_article_sent_tokens,
                                                                                           lcs_paths_list=lcs_paths_list,
                                                                                           article_lcs_paths_list=highlight_smooth_article_lcs_paths_list,
                                                                                             fusion_locations=original_coref_chains
                                                                                             )
                                                                                           # gt_similar_source_indices_list=[list(range(len(ssi)))],
                                                                                           # gt_article_lcs_paths_list=selected_article_lcs_paths)
                            highlight_html_other += highlighted_html + '<br>'
                            highlight_html_other += '%.3f %.3f %.3f <br> %.3f %.3f %.3f <br>' % (my_r1, my_r2, my_rl, o_r1, o_r2, o_rl)


                        if FLAGS.attn_vis and example_idx < 200:
                            self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists,
                                                   best_hyp.p_gens,
                                                   attn_vis_idx)  # write info to .json file for visualization tool
                            attn_vis_idx += 1

                        if len(final_decoded_words) >= 100:
                            break

                    sources_lens.append(len(my_sources))
                    gt_ssi_list, gt_alp_list = util.replace_empty_ssis(groundtruth_similar_source_indices_list, raw_article_sents, sys_alp_list=groundtruth_article_lcs_paths_list)
                    highlight_html_gt = '<u>Reference Summary</u><br><br>'
                    for ssi_idx, ssi in enumerate(gt_ssi_list):
                        selected_article_lcs_paths = gt_alp_list[ssi_idx]
                        try:
                            ssi, selected_article_lcs_paths = util.make_ssi_chronological(ssi, selected_article_lcs_paths)
                        except:
                            util.print_vars(ssi, example_idx, selected_article_lcs_paths)
                            raise
                        selected_raw_article_sents = util.reorder(raw_article_sents, ssi)

                        if example_idx < 100 or (example_idx >= 2000 and example_idx < 2100) or is_better or is_worse:
                            min_matched_tokens = 2
                            selected_article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in selected_raw_article_sents]
                            highlight_summary_sent_tokens = [groundtruth_summ_sent_tokens[ssi_idx]]
                            highlight_ssi_list, lcs_paths_list, highlight_article_lcs_paths_list, highlight_smooth_article_lcs_paths_list = ssi_functions.get_simple_source_indices_list(
                                highlight_summary_sent_tokens,
                                selected_article_sent_tokens, 2, min_matched_tokens)
                            highlighted_html = ssi_functions.html_highlight_sents_in_article(highlight_summary_sent_tokens,
                                                                                           highlight_ssi_list,
                                                                                             selected_article_sent_tokens,
                                                                                           lcs_paths_list=lcs_paths_list,
                                                                                           article_lcs_paths_list=highlight_smooth_article_lcs_paths_list,
                                                                                             fusion_locations=original_coref_chains
                                                                                                     )
                            highlight_html_gt += highlighted_html + '<br>'

                    if example_idx < 100 or (example_idx >= 2000 and example_idx < 2100):
                        self.write_for_human(raw_article_sents, groundtruth_summ_sents, final_decoded_sent_tokens, example_idx, already_sent_split=True)
                        # highlight_html_total = ssi_functions.put_html_in_two_columns(highlight_html_total + highlight_html_tagged, highlight_html_gt)
                        highlight_html_total = highlight_html_total + highlight_html_gt
                        ssi_functions.write_highlighted_html(highlight_html_total, self._highlight_dir, example_idx)

                    if is_better:
                        highlight_html_compare = highlight_html_total + highlight_html_other + highlight_html_gt
                        ssi_functions.write_highlighted_html(highlight_html_compare, self._better_dir, better_idx)
                        better_idx += 1

                    if is_worse:
                        highlight_html_compare = highlight_html_total + highlight_html_other + highlight_html_gt
                        ssi_functions.write_highlighted_html(highlight_html_compare, self._worse_dir, worse_idx)
                        worse_idx += 1

                    # if example_idx % 100 == 0:
                    #     attn_dir = os.path.join(self._decode_dir, 'attn_vis_data')
                    #     attn_selections.process_attn_selections(attn_dir, self._decode_dir, self._vocab)

                    final_decoded_sents = [' '.join(sent) for sent in final_decoded_sent_tokens]
                    rouge_functions.write_for_rouge(groundtruth_summ_sents, final_decoded_sents, example_idx, self._rouge_ref_dir, self._rouge_dec_dir, log=False) # write ref summary and decoded summary to file, to eval with pyrouge later
                    # if FLAGS.attn_vis:
                    #     self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens, example_idx) # write info to .json file for visualization tool
                example_idx += 1 # this is how many examples we've decoded

        if FLAGS.worker_idx == -1 or FLAGS.worker_idx == FLAGS.num_workers - 1:     # Only perform ROUGE eval if this is the last worker (or if we are not using workers)
            logging.info("Decoder has finished reading dataset for single_pass.")
            logging.info("Output has been saved in %s and %s.", self._rouge_ref_dir, self._rouge_dec_dir)

            if FLAGS.worker_idx == FLAGS.num_workers - 1:       # Wait for other workers to finish before evaluating
                self.write_worker_signal(True)
                while not self.check_workers_done():
                    print('Waiting on other workers. Sleeping for 30s...')
                    time.sleep(30)

            # print (np.median(sources_lens))
            # print('perc words tagged', np.median(perc_words_tagged))
            #
            # print('Evaluating word tagging F1 score...')
            # suffix = util.word_tag_eval(all_gt_word_tags, all_sys_word_tags)
            # print (suffix)
            # rouge_functions.word_f1_log(self._decode_dir, suffix)

            if FLAGS.corr_tests:
                logging.info('Performing correlation between input features and output ROUGE scores...')
                print(corr_features)
                for output_key in corr_outputs.keys():
                    for feature_key in corr_features.keys():
                        coeff, pval = pearsonr(corr_features[feature_key], corr_outputs[output_key])
                        logging.info('%s %s:\t%.3f\t%.6f' % (output_key, feature_key, coeff, pval))

            if len(os.listdir(self._rouge_ref_dir)) != 0 and not FLAGS.word_eval_only:
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
                # print('Calculating bert score')
                bleu = nltk.translate.bleu_score.corpus_bleu([[ref] for ref in refs], cands) * 100
                # print(bleu)
                bert_p, bert_r, bert_f = bert_score.score(cands, refs, lang="en", verbose=False, batch_size=8,
                                                          model_type='bert-base-uncased')
                # print(bert_p, bert_r, bert_f)
                avg_len = np.mean(lens)
                bert_p = np.mean(bert_p.cpu().numpy()) * 100
                bert_r = np.mean(bert_r.cpu().numpy()) * 100
                bert_f = np.mean(bert_f.cpu().numpy()) * 100
                # print(bert_p)
                sheets_results_contents = sheets_results_text.strip().split('\t')
                new_results = ['%.2f' % avg_len] + sheets_results_contents + ['%.2f' % bert_p, '%.2f' % bert_r,
                                                                              '%.2f' % bert_f, '%.2f' % bleu]
                print('\t'.join(new_results))
                new_sheets_results_file = os.path.join(self._decode_dir, 'bert_sheets_results.txt')
                with open(new_sheets_results_file, 'w') as f:
                    f.write('\t'.join(new_results))
        else:
            self.write_worker_signal(True)
            print('Worker %d has finished his work.' % FLAGS.worker_idx)



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

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens, ex_index, ssi=None):
        """Write some data to json file, which can be read into the in-browser attention visualizer tool:
            https://github.com/abisee/attn_vis

        Args:
            article: The original article string.
            abstract: The human (correct) abstract string.
            attn_dists: List of arrays; the attention distributions.
            decoded_words: List of strings; the words of the generated summary.
            p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
        """
        article_lst = article.split() # list of words
        decoded_lst = decoded_words # list of decoded words
        to_write = {
                'article_lst': [make_html_safe(t) for t in article_lst],
                'decoded_lst': [make_html_safe(t) for t in decoded_lst],
                'abstract_str': make_html_safe(abstract),
                'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        if ssi is not None:
            to_write['ssi'] = ssi
        util.create_dirs(os.path.join(self._decode_dir, 'attn_vis_data'))
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data', '%06d.json' % ex_index)
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        # logging.info('Wrote visualization data to %s', output_fname)

    def calc_importance_features(self, data_path, hps, model_save_path, docs_desired):
        """Calculate sentence-level features and save as a dataset"""
        data_path_filter_name = os.path.basename(data_path)
        if 'train' in data_path_filter_name:
            data_split = 'train'
        elif 'val' in data_path_filter_name:
            data_split = 'val'
        elif 'test' in data_path_filter_name:
            data_split = 'test'
        else:
            data_split = 'feats'
        if 'cnn-dailymail' in data_path:
            inst_per_file = 1000
        else:
            inst_per_file = 1
        filelist = glob.glob(data_path)
        num_documents_desired = docs_desired
        pbar = tqdm(initial=0, total=num_documents_desired)

        instances = []
        sentences = []
        counter = 0
        doc_counter = 0
        file_counter = 0
        while True:
            batch = self._batcher.next_batch()	# 1 example repeated across batch
            if doc_counter >= num_documents_desired:
                save_path = os.path.join(model_save_path, data_split + '_%06d'%file_counter)
                with open(save_path, 'wb') as f:
                    pickle.dump(instances, f)
                print(('Saved features at %s' % save_path))
                return

            if batch is None: # finished decoding dataset in single_pass mode
                raise Exception('We havent reached the num docs desired (%d), instead we reached (%d)' % (num_documents_desired, doc_counter))


            batch_enc_states, _ = self._model.run_encoder(self._sess, batch)
            for batch_idx, enc_states in enumerate(batch_enc_states):
                art_oovs = batch.art_oovs[batch_idx]
                all_original_abstracts_sents = batch.all_original_abstracts_sents[batch_idx]

                tokenizer = Tokenizer('english')
                # List of lists of words
                enc_sentences, enc_tokens = batch.tokenized_sents[batch_idx], batch.word_ids_sents[batch_idx]
                enc_sent_indices = importance_features.get_sent_indices(enc_sentences, batch.doc_indices[batch_idx])
                enc_sentences_str = [' '.join(sent) for sent in enc_sentences]

                sent_representations_separate = importance_features.get_separate_enc_states(self._model, self._sess, enc_sentences, self._vocab, hps)

                sent_indices = enc_sent_indices
                sent_reps = importance_features.get_importance_features_for_article(
                    enc_states, enc_sentences, sent_indices, tokenizer, sent_representations_separate)
                y, y_hat = importance_features.get_ROUGE_Ls(art_oovs, all_original_abstracts_sents, self._vocab, enc_tokens)
                binary_y = importance_features.get_best_ROUGE_L_for_each_abs_sent(art_oovs, all_original_abstracts_sents, self._vocab, enc_tokens)
                for rep_idx, rep in enumerate(sent_reps):
                    rep.y = y[rep_idx]
                    rep.binary_y = binary_y[rep_idx]

                for rep_idx, rep in enumerate(sent_reps):
                    # Keep all sentences with importance above threshold. All others will be kept with a probability of prob_to_keep
                    if FLAGS.importance_fn == 'svr':
                        instances.append(rep)
                        sentences.append(sentences)
                        counter += 1 # this is how many examples we've decoded
            doc_counter += len(batch_enc_states)
            pbar.update(len(batch_enc_states))



def decode_example(sess, model, vocab, batch, counter, hps):
    # Run beam search to get best Hypothesis
    best_hyp = beam_search.run_beam_search(sess, model, vocab, batch, counter, hps)

    # Extract the output ids from the hypothesis and convert back to words
    output_ids = [int(t) for t in best_hyp.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

    # Remove the [STOP] token from decoded_words, if necessary
    try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
    except ValueError:
        decoded_words = decoded_words
    decoded_output = ' '.join(decoded_words) # single string
    return decoded_words, decoded_output, best_hyp


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
    if FLAGS.tag_tokens:
        dirname += '_%.02f' % FLAGS.tag_threshold
    dirname += "_%imaxenc_%imindec_%imaxdec" % (FLAGS.max_enc_steps, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
