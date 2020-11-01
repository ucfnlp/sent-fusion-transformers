import textwrap as tw
import PIL
import itertools
import util
from util import get_similarity, rouge_l_similarity
import importance_features
import dill
import time
import random
import numpy as np
import os
import data
from absl import flags
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS

def convert_to_word_level(mmr_for_sentences, enc_tokens):
    num_tokens = len(util.flatten_list_of_lists(enc_tokens))
    mmr = np.ones([num_tokens], dtype=float) / num_tokens
    # Calculate how much for each word in source
    word_idx = 0
    for sent_idx in range(len(enc_tokens)):
        mmr_for_words = np.full([len(enc_tokens[sent_idx])], mmr_for_sentences[sent_idx])
        mmr[word_idx:word_idx + len(mmr_for_words)] = mmr_for_words
        word_idx += len(mmr_for_words)
    return mmr

def plot_importances(article_sents, importances, abstracts_text, save_location=None, save_name=None):
    plt.ioff()
    sents_per_figure = 40
    max_importance = np.max(importances)
    chunked_sents = util.chunks(article_sents, sents_per_figure)
    chunked_importances = util.chunks(importances, sents_per_figure)

    for chunk_idx in range(len(chunked_sents)):
        my_article_sents = chunked_sents[chunk_idx]
        my_importances = chunked_importances[chunk_idx]

        if len(my_article_sents) < sents_per_figure:
            my_article_sents += [''] * (sents_per_figure-len(my_article_sents))
            my_importances = np.concatenate([my_importances, np.zeros([sents_per_figure-len(my_importances)])])

        y_pos = np.arange(len(my_article_sents))
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(left=0.9, top=1.0, bottom=0.03, right=1.0)
        ax1.barh(y_pos, my_importances, align='center',
                 color='green', ecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(my_article_sents)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Performance')
        ax1.set_title('How fast do you want to go today?')
        ax1.set_xlim(right=max_importance)

        fig.set_size_inches(18.5, 10.5)
        plt.savefig(os.path.join(save_location, save_name + '_' + str(chunk_idx) + '.jpg'))
        plt.close(fig)

    plt.figure()
    fig_txt = tw.fill(tw.dedent(abstracts_text), width=80)
    plt.figtext(0.5, 0.5, fig_txt, horizontalalignment='center',
                fontsize=9, multialignment='left',
                bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                          ec="0.5", pad=0.5, alpha=1), fontweight='bold')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(save_location, save_name + '_' + str(chunk_idx+1) + '.jpg'))
    plt.close(fig)

def calc_mmr_from_sim_and_imp(similarity, importances):
    new_mmr =  FLAGS.lambda_val*importances - (1-FLAGS.lambda_val)*similarity
    new_mmr = np.maximum(new_mmr, 0)
    return new_mmr

def mute_all_except_top_k(array, k):
    num_reservoirs_still_full = np.sum(array > 0)
    if num_reservoirs_still_full < k:
        selected_indices = np.nonzero(array)
    else:
        selected_indices = array.argsort()[::-1][:k]
    res = np.zeros_like(array, dtype=float)
    for selected_idx in selected_indices:
        if FLAGS.retain_mmr_values:
            res[selected_idx] = array[selected_idx]
        else:
            res[selected_idx] = 1.
    return res

def get_tokens_for_human_summaries(batch, vocab):
    art_oovs = batch.art_oovs[0]
    def get_all_summ_tokens(all_summs):
        return [get_summ_tokens(summ) for summ in all_summs]
    def get_summ_tokens(summ):
        summ_tokens = [get_sent_tokens(sent) for sent in summ]
        return list(itertools.chain.from_iterable(summ_tokens))     # combines all sentences into one list of tokens for summary
    def get_sent_tokens(sent):
        words = sent.split()
        return data.abstract2ids(words, vocab, art_oovs)
    human_summaries = batch.all_original_abstracts_sents[0]
    all_summ_tokens = get_all_summ_tokens(human_summaries)
    return all_summ_tokens

def get_svr_importances(enc_states, enc_sentences, enc_sent_indices, svr_model, sent_representations_separate):
    sent_indices = enc_sent_indices
    sent_reps = importance_features.get_importance_features_for_article(
        enc_states, enc_sentences, sent_indices, sent_representations_separate)
    features_list = importance_features.get_features_list(False)
    x = importance_features.features_to_array(sent_reps, features_list)
    if FLAGS.importance_fn == 'svr':
        importances = svr_model.predict(x)
    else:
        importances = svr_model.decision_function(x)
    return importances

def get_tfidf_importances(raw_article_sents):
    tfidf_model_path = os.path.join(FLAGS.actual_log_root, 'tfidf_vectorizer', FLAGS.original_dataset_name + '.dill')

    while True:
        try:
            with open(tfidf_model_path, 'rb') as f:
                tfidf_vectorizer = dill.load(f)
            break
        except (EOFError, KeyError):
            time.sleep(random.randint(3,6))
            continue
    sent_reps = tfidf_vectorizer.transform(raw_article_sents)
    cluster_rep = np.mean(sent_reps, axis=0)
    similarity_matrix = cosine_similarity(sent_reps, cluster_rep)
    return np.squeeze(similarity_matrix)

def get_importances(model, batch, enc_states, vocab, sess, hps):
    if FLAGS.pg_mmr:
        enc_sentences, enc_tokens = batch.tokenized_sents[0], batch.word_ids_sents[0]
        if FLAGS.importance_fn == 'oracle':
            human_tokens = get_tokens_for_human_summaries(batch, vocab)     # list (of 4 human summaries) of list of token ids
            metric = 'recall'
            importances_hat = rouge_l_similarity(enc_tokens, human_tokens, vocab, metric=metric)
        elif FLAGS.importance_fn == 'svr':
            if FLAGS.importance_fn == 'svr':
                with open(os.path.join(FLAGS.actual_log_root, 'svr.pickle'), 'rb') as f:
                    svr_model = pickle.load(f)
            enc_sent_indices = importance_features.get_sent_indices(enc_sentences, batch.doc_indices[0])
            sent_representations_separate = importance_features.get_separate_enc_states(model, sess, enc_sentences, vocab, hps)
            importances_hat = get_svr_importances(enc_states[0], enc_sentences, enc_sent_indices, svr_model, sent_representations_separate)
        elif FLAGS.importance_fn == 'tfidf':
            importances_hat = get_tfidf_importances(batch.raw_article_sents[0])
        importances = util.special_squash(importances_hat)
    else:
        importances = None
    return importances