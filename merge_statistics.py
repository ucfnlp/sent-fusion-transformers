# from pathos.multiprocessing import ProcessingPool as Pool
import itertools
import os
from tqdm import tqdm
import numpy as np
from absl import flags
from absl import app
import pickle
import util
import sys
import glob
import data
from scipy.stats.stats import pearsonr
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'all', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'val', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens'
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]
min_matched_tokens = 1
tfidf_vec_path = 'data/tfidf/' + 'all' + '_tfidf_vec_5.pkl'
bin_values = [x / 100. for x in list(range(100))]
pretty_dataset_names = {'cnn_dm': 'CNN/DM', 'xsum': 'XSum', 'duc_2004': 'DUC-04'}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20

util.create_dirs('stuff/plots')

plot_data_file = os.path.join('stuff/plots', FLAGS.dataset_name + '_' + FLAGS.dataset_split + '.pkl')
plot_file = os.path.join('stuff/plots', FLAGS.dataset_name + '_' + FLAGS.dataset_split + '.pdf')

def plot_histograms(all_list_of_hist_pairs):
    nrows = len(all_list_of_hist_pairs)
    ncols = len(all_list_of_hist_pairs[0])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    fig.set_size_inches(10, 5)
    fig.subplots_adjust(wspace=0.075, hspace=0.05)
    for row_idx in range(axes.shape[0]):
        for col_idx in range(axes.shape[1]):
            ax = axes[row_idx, col_idx]
            plot_histogram(ax, row_idx, col_idx, **all_list_of_hist_pairs[row_idx][col_idx])
    pp = PdfPages(plot_file)
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    plt.show()
    pp.close()

font = {
        'size': 20,
        }

def plot_histogram(ax, row_idx, col_idx, lst=None, num_bins=None, start_at_0=False, pdf=False, cutoff_std=4, log=False, max_val=None, y_label=None, x_label=None, x_lim=None, y_lim=None, legend_labels=None):
    def plot(my_lst, translucent=False, legend_label=None):
        alpha = 0.5 if translucent else 1
        histtype = 'stepfilled'
        if max_val is None:
            my_max_val = np.mean(my_lst) + cutoff_std*np.std(my_lst)
        else:
            my_max_val = max_val
        # bins = 100 if normalized else list(range(min(my_lst), max(my_lst) + 2)) if not start_at_0 else list(range(max(my_lst) + 2))
        bins = num_bins if num_bins is not None else list(range(int(min(my_lst)), int(my_max_val))) if not start_at_0 else list(range(int(my_max_val)))
        ax.hist(my_lst, bins=bins, density=pdf, alpha=alpha, histtype=histtype, edgecolor='black', log=log, label=legend_label)
        if row_idx == 0:
            ax.legend()
        if y_label is not None:
            ax.set_ylabel(pretty_dataset_names[y_label])
        if row_idx == 2:
            ax.set_xlabel(x_label)
        else:
            ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_yticklabels([])
        if y_lim is not None:
            ax.set_ylim(top=y_lim)
        if x_lim is not None:
            ax.set_xlim(right=x_lim)
        # nbins = len(ax.get_xticklabels())
        if row_idx > 0:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))

    for lst_idx, my_lst in enumerate(lst):
        if legend_labels is not None:
            legend_label = legend_labels[lst_idx]
        plot(my_lst, translucent=True, legend_label=legend_label)

    # fig, ax1 = plt.subplots(nrows=1)
    # fig.set_size_inches(6, 2)
    # varname = util.varname(lst)[0]
    # if type(lst[0]) == list:
    #     for my_lst in lst:
    #         plot(my_lst, translucent=True)
    #
    # pp = PdfPages(os.path.join('stuff/plots', FLAGS.dataset_name + '_' + varname + '.pdf'))
    # plt.savefig(pp, format='pdf',bbox_inches='tight')
    # plt.show()
    # pp.close()

# def plot_histogram(lst, num_bins=None, start_at_0=False, pdf=False, cutoff_std=4, log=False, max_val=None):
#     def plot(my_lst, translucent=False):
#         alpha = 0.5 if translucent else 1
#         histtype = 'stepfilled'
#         if max_val is None:
#             my_max_val = np.mean(my_lst) + cutoff_std*np.std(my_lst)
#         else:
#             my_max_val = max_val
#         # bins = 100 if normalized else list(range(min(my_lst), max(my_lst) + 2)) if not start_at_0 else list(range(max(my_lst) + 2))
#         bins = num_bins if num_bins is not None else list(range(int(min(my_lst)), int(my_max_val))) if not start_at_0 else list(range(int(my_max_val)))
#         plt.hist(my_lst, bins=bins, density=pdf, alpha=alpha, histtype=histtype, edgecolor='black', log=log)
#
#     fig, ax1 = plt.subplots(nrows=1)
#     fig.set_size_inches(6, 2)
#     varname = util.varname(lst)[0]
#     if type(lst[0]) == list:
#         for my_lst in lst:
#             plot(my_lst, translucent=True)
#     else:
#         plot(lst)
#
#     pp = PdfPages(os.path.join('stuff/plots', FLAGS.dataset_name + '_' + varname + '.pdf'))
#     plt.savefig(pp, format='pdf',bbox_inches='tight')
#     plt.show()
#     pp.close()


def plot_positions(primary_pos, secondary_pos, all_pos):
    print('Sentence positions (primary (mean, median), secondary (mean, median), all (mean, median)) : ', np.mean(primary_pos), np.median(primary_pos), np.mean(secondary_pos), np.median(secondary_pos), np.mean(all_pos), np.median(all_pos))
    hist_pos_primary = np.histogram(primary_pos, bins=max(primary_pos)+1)
    hist_pos_secondary = np.histogram(secondary_pos, bins=max(secondary_pos)+1)
    hist_pos_all = np.histogram(all_pos, bins=max(all_pos)+1)
    print('Histogram of positions primary:', util.hist_as_pdf_str(hist_pos_primary))
    print('Histogram of positions secondary:', util.hist_as_pdf_str(hist_pos_secondary))
    print('Histogram of positions all:', util.hist_as_pdf_str(hist_pos_all))

    plot_histogram(primary_pos)
    plot_histogram(secondary_pos)
    plot_histogram(all_pos)

def get_integral_values_for_histogram(orig_val, rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents):
    if FLAGS.dataset_name == 'duc_2004':
        val = rel_sent_indices[orig_val]
        num_sents_total = doc_sent_lens[doc_sent_indices[orig_val]]
    else:
        val = orig_val
        num_sents_total = len(raw_article_sents)
    norm = val*1./num_sents_total
    next_norm = (val+1)*1./num_sents_total
    vals_to_add = [bin_val for bin_val in bin_values if bin_val >= norm and bin_val < next_norm]
    return vals_to_add


def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)


    if FLAGS.dataset_name == 'all':
        dataset_names = ['cnn_dm', 'xsum', 'duc_2004']
    else:
        dataset_names = [FLAGS.dataset_name]

    # if not os.path.exists(plot_data_file):
    if True:
        all_lists_of_histogram_pairs = []
        for dataset_name in dataset_names:
            FLAGS.dataset_name = dataset_name

            if dataset_name == 'duc_2004':
                dataset_splits = ['test']
            elif FLAGS.dataset_split == 'all':
                dataset_splits = ['test', 'val', 'train']
            else:
                dataset_splits = [FLAGS.dataset_split]

            ssi_list = []
            for dataset_split in dataset_splits:

                ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name, dataset_split + '_ssi.pkl')

                with open(ssi_path, 'rb') as f:
                    ssi_list.extend(pickle.load(f))

                if FLAGS.dataset_name == 'duc_2004':
                    for abstract_idx in [1,2,3]:
                        ssi_path = os.path.join(ssi_dir, FLAGS.dataset_name, dataset_split + '_ssi_' + str(abstract_idx) + '.pkl')
                        with open(ssi_path, 'rb') as f:
                            temp_ssi_list = pickle.load(f)
                        ssi_list.extend(temp_ssi_list)

            ssi_2d = util.flatten_list_of_lists(ssi_list)

            num_extracted = [len(ssi) for ssi in util.flatten_list_of_lists(ssi_list)]
            hist_num_extracted = np.histogram(num_extracted, bins=6, range=(0,5))
            print(hist_num_extracted)
            print('Histogram of number of sentences merged: ' + util.hist_as_pdf_str(hist_num_extracted))

            distances = [abs(ssi[0]-ssi[1]) for ssi in ssi_2d if len(ssi) >= 2]
            print('Distance between sentences (mean, median): ', np.mean(distances), np.median(distances))
            hist_dist = np.histogram(distances, bins=max(distances))
            print('Histogram of distances: ' + util.hist_as_pdf_str(hist_dist))

            summ_sent_idx_to_number_of_source_sents = [[], [], [], [], [], [], [], [], [], []]
            for ssi in ssi_list:
                for summ_sent_idx, source_indices in enumerate(ssi):
                    if len(source_indices) == 0 or summ_sent_idx >= len(summ_sent_idx_to_number_of_source_sents):
                        continue
                    num_sents = len(source_indices)
                    if num_sents > 2:
                        num_sents = 2
                    summ_sent_idx_to_number_of_source_sents[summ_sent_idx].append(num_sents)
            print ("Number of source sents for summary sentence indices (Is the first summary sent more likely to match with a singleton or a pair?):")
            for summ_sent_idx, list_of_numbers_of_source_sents in enumerate(summ_sent_idx_to_number_of_source_sents):
                if len(list_of_numbers_of_source_sents) == 0:
                    percent_singleton = 0.
                else:
                    percent_singleton = list_of_numbers_of_source_sents.count(1) * 1. / len(list_of_numbers_of_source_sents)
                    percent_pair = list_of_numbers_of_source_sents.count(2) * 1. / len(list_of_numbers_of_source_sents)
                print (str(percent_singleton) + '\t',)
            print ('')
            for summ_sent_idx, list_of_numbers_of_source_sents in enumerate(summ_sent_idx_to_number_of_source_sents):
                if len(list_of_numbers_of_source_sents) == 0:
                    percent_pair = 0.
                else:
                    percent_singleton = list_of_numbers_of_source_sents.count(1) * 1. / len(list_of_numbers_of_source_sents)
                    percent_pair = list_of_numbers_of_source_sents.count(2) * 1. / len(list_of_numbers_of_source_sents)
                print (str(percent_pair) + '\t',)
            print ('')

            primary_pos = [ssi[0] for ssi in ssi_2d if len(ssi) >= 1]
            secondary_pos = [ssi[1] for ssi in ssi_2d if len(ssi) >= 2]
            all_pos = [max(ssi) for ssi in ssi_2d if len(ssi) >= 1]

            # if FLAGS.dataset_name != 'duc_2004':
            #     plot_positions(primary_pos, secondary_pos, all_pos)

            if FLAGS.dataset_split == 'all':
                glob_string = '*.bin'
            else:
                glob_string = dataset_splits[0]

            print('Loading TFIDF vectorizer')
            try:
                with open(tfidf_vec_path, 'rb') as f:
                    tfidf_vectorizer = pickle.load(f)
            except:
                with open(tfidf_vec_path, 'rb') as f:
                    tfidf_vectorizer = pickle.load(f, encoding='latin1')

            source_dir = os.path.join(data_dir, FLAGS.dataset_name)
            source_files = sorted(glob.glob(source_dir + '/' + glob_string + '*'))

            total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
            example_generator = data.example_generator(source_dir + '/' + glob_string + '*', True, False,
                                                       should_check_valid=False)

            all_possible_singles = 0
            all_possible_pairs = [0]
            all_filtered_pairs = 0
            all_all_combinations = 0
            all_ssi_pairs = [0]
            ssi_pairs_with_shared_coref = [0]
            ssi_pairs_with_shared_word = [0]
            ssi_pairs_with_either_coref_or_word = [0]
            all_pairs_with_shared_coref = [0]
            all_pairs_with_shared_word = [0]
            all_pairs_with_either_coref_or_word = [0]
            actual_total = [0]
            rel_positions_primary = []
            rel_positions_secondary = []
            rel_positions_all = []
            sent_lens = []
            all_sent_lens = []
            all_pos = []
            y = []
            normalized_positions_primary = []
            normalized_positions_secondary = []
            all_normalized_positions_primary = []
            all_normalized_positions_secondary = []
            normalized_positions_singles = []
            normalized_positions_pairs_first = []
            normalized_positions_pairs_second = []
            primary_pos_duc = []
            secondary_pos_duc = []
            all_pos_duc = []
            all_distances = []
            distances_duc = []
            tfidf_similarities = []
            all_tfidf_similarities = []
            average_mmrs = []
            all_average_mmrs = []
            all_num_tagged_tokens = []

            for example_idx, example in enumerate(tqdm(example_generator, total=total)):
            # def process(example_idx_example):
            #     # print '0'
            #     example = example_idx_example
                if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                    break
                raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices, article_lcs_paths_list = util.unpack_tf_example(
                    example, names_to_types)
                article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
                article_text = ' '.join(raw_article_sents)
                groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
                if doc_indices is None:
                    doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
                doc_indices = [int(doc_idx) for doc_idx in doc_indices]
                rel_sent_indices, doc_sent_indices, doc_sent_lens = util.get_rel_sent_indices(doc_indices, article_sent_tokens)
                groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)

                # sent_term_matrix = util.get_doc_substituted_tfidf_matrix(tfidf_vectorizer, raw_article_sents, article_text)
                # sents_similarities = util.cosine_similarity(sent_term_matrix, sent_term_matrix)
                # importances = util.special_squash(util.get_tfidf_importances(tfidf_vectorizer, raw_article_sents))
                #
                # if FLAGS.dataset_name == 'duc_2004':
                #     first_k_indices = util.get_indices_of_first_k_sents_of_each_article(rel_sent_indices, FLAGS.first_k)
                # else:
                #     first_k_indices = [idx for idx in range(len(raw_article_sents))]
                # article_indices = list(range(len(raw_article_sents)))
                #
                # possible_pairs = [x for x in list(itertools.combinations(article_indices, 2))]  # all pairs
                # # # # filtered_possible_pairs = preprocess_for_lambdamart_no_flags.filter_pairs_by_criteria(raw_article_sents, possible_pairs, corefs)
                # # if FLAGS.dataset_name == 'duc_2004':
                # #     filtered_possible_pairs = [x for x in list(itertools.combinations(first_k_indices, 2))]  # all pairs
                # # else:
                # #     filtered_possible_pairs = preprocess_for_lambdamart_no_flags.filter_pairs_by_sent_position(possible_pairs)
                # # # removed_pairs = list(set(possible_pairs) - set(filtered_possible_pairs))
                # # possible_singles = [(i,) for i in range(len(raw_article_sents))]
                # # all_combinations = filtered_possible_pairs + possible_singles
                # #
                # # all_possible_singles += len(possible_singles)
                # # all_possible_pairs[0] += len(possible_pairs)
                # # all_filtered_pairs += len(filtered_possible_pairs)
                # # all_all_combinations += len(all_combinations)
                #
                # # for ssi in groundtruth_similar_source_indices_list:
                # #     if len(ssi) > 0:
                # #         idx = rel_sent_indices[ssi[0]]
                # #         rel_positions_primary.append(idx)
                # #         rel_positions_all.append(idx)
                # #     if len(ssi) > 1:
                # #         idx = rel_sent_indices[ssi[1]]
                # #         rel_positions_secondary.append(idx)
                # #         rel_positions_all.append(idx)
                # #
                # #
                # #
                #
                # # coref_pairs = preprocess_for_lambdamart_no_flags.get_coref_pairs(corefs)
                # # # DO OVER LAP PAIRS BETTER
                # # overlap_pairs = preprocess_for_lambdamart_no_flags.filter_by_overlap(article_sent_tokens, possible_pairs)
                # # either_coref_or_word = list(set(list(coref_pairs) + overlap_pairs))
                # #
                # # for ssi in groundtruth_similar_source_indices_list:
                # #     if len(ssi) == 2:
                # #         all_ssi_pairs[0] += 1
                # #         do_share_coref = ssi in coref_pairs
                # #         do_share_words = ssi in overlap_pairs
                # #         if do_share_coref:
                # #             ssi_pairs_with_shared_coref[0] += 1
                # #         if do_share_words:
                # #             ssi_pairs_with_shared_word[0] += 1
                # #         if do_share_coref or do_share_words:
                # #             ssi_pairs_with_either_coref_or_word[0] += 1
                # # all_pairs_with_shared_coref[0] += len(coref_pairs)
                # # all_pairs_with_shared_word[0] += len(overlap_pairs)
                # # all_pairs_with_either_coref_or_word[0] += len(either_coref_or_word)
                #
                # if FLAGS.dataset_name == 'duc_2004':
                #     primary_pos_duc.extend([rel_sent_indices[ssi[0]] for ssi in groundtruth_similar_source_indices_list if len(ssi) >= 1])
                #     secondary_pos_duc.extend([rel_sent_indices[ssi[1]] for ssi in groundtruth_similar_source_indices_list if len(ssi) >= 2])
                #     all_pos_duc.extend([max([rel_sent_indices[sent_idx] for sent_idx in ssi]) for ssi in groundtruth_similar_source_indices_list if len(ssi) >= 1])
                #
                # for ssi in groundtruth_similar_source_indices_list:
                #     for sent_idx in ssi:
                #         sent_lens.append(len(article_sent_tokens[sent_idx]))
                #     if len(ssi) >= 1:
                #         orig_val = ssi[0]
                #         vals_to_add = get_integral_values_for_histogram(orig_val, rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents)
                #         normalized_positions_primary.extend(vals_to_add)
                #     if len(ssi) >= 2:
                #         orig_val = ssi[1]
                #         vals_to_add = get_integral_values_for_histogram(orig_val, rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents)
                #         normalized_positions_secondary.extend(vals_to_add)
                #
                #         if FLAGS.dataset_name == 'duc_2004':
                #             distances_duc.append(abs(rel_sent_indices[ssi[1]] - rel_sent_indices[ssi[0]]))
                #
                #         tfidf_similarities.append(sents_similarities[ssi[0], ssi[1]])
                #         average_mmrs.append((importances[ssi[0]] + importances[ssi[1]])/2)
                #
                # for ssi in groundtruth_similar_source_indices_list:
                #     if len(ssi) == 1:
                #         orig_val = ssi[0]
                #         vals_to_add = get_integral_values_for_histogram(orig_val, rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents)
                #         normalized_positions_singles.extend(vals_to_add)
                #     if len(ssi) >= 2:
                #         if doc_sent_indices[ssi[0]] != doc_sent_indices[ssi[1]]:
                #             continue
                #         orig_val_first = min(ssi[0], ssi[1])
                #         vals_to_add = get_integral_values_for_histogram(orig_val_first, rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents)
                #         normalized_positions_pairs_first.extend(vals_to_add)
                #         orig_val_second = max(ssi[0], ssi[1])
                #         vals_to_add = get_integral_values_for_histogram(orig_val_second, rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents)
                #         normalized_positions_pairs_second.extend(vals_to_add)
                #
                # # all_normalized_positions_primary.extend(util.flatten_list_of_lists([get_integral_values_for_histogram(single[0], rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents) for single in possible_singles]))
                # # all_normalized_positions_secondary.extend(util.flatten_list_of_lists([get_integral_values_for_histogram(pair[1], rel_sent_indices, doc_sent_indices, doc_sent_lens, raw_article_sents) for pair in possible_pairs]))
                # all_sent_lens.extend([len(sent) for sent in article_sent_tokens])
                # all_distances.extend([abs(rel_sent_indices[pair[1]] - rel_sent_indices[pair[0]]) for pair in possible_pairs])
                # all_tfidf_similarities.extend([sents_similarities[pair[0], pair[1]] for pair in possible_pairs])
                # all_average_mmrs.extend([(importances[pair[0]] + importances[pair[1]])/2 for pair in possible_pairs])
                #
                # # if FLAGS.dataset_name == 'duc_2004':
                # #     rel_pos_single = [rel_sent_indices[single[0]] for single in possible_singles]
                # #     rel_pos_pair = [[rel_sent_indices[pair[0]], rel_sent_indices[pair[1]]] for pair in possible_pairs]
                # #     all_pos.extend(rel_pos_single)
                # #     all_pos.extend([max(pair) for pair in rel_pos_pair])
                # # else:
                # #     all_pos.extend(util.flatten_list_of_lists(possible_singles))
                # #     all_pos.extend([max(pair) for pair in possible_pairs])
                # # y.extend([1 if single in groundtruth_similar_source_indices_list else 0 for single in possible_singles])
                # # y.extend([1 if pair in groundtruth_similar_source_indices_list else 0 for pair in possible_pairs])
                #
                # # actual_total[0] += 1

                num_tagged_tokens = sum([sum([len(c) for c in b]) for b in article_lcs_paths_list])
                all_num_tagged_tokens.append(num_tagged_tokens)


            # # p = Pool(144)
            # # list(tqdm(p.imap(process, example_generator), total=total))
            #
            # # print 'Possible_singles\tPossible_pairs\tFiltered_pairs\tAll_combinations: \n%.2f\t%.2f\t%.2f\t%.2f' % (all_possible_singles*1./actual_total, \
            # #     all_possible_pairs*1./actual_total, all_filtered_pairs*1./actual_total, all_all_combinations*1./actual_total)
            # #
            # # # print 'Relative positions of groundtruth source sentences in document:\nPrimary\tSecondary\tBoth\n%.2f\t%.2f\t%.2f' % (np.mean(rel_positions_primary), np.mean(rel_positions_secondary), np.mean(rel_positions_all))
            # #
            # # print 'SSI Pair statistics:\nShare_coref\tShare_word\tShare_either\n%.2f\t%.2f\t%.2f' \
            # #       % (ssi_pairs_with_shared_coref[0]*100./all_ssi_pairs[0], ssi_pairs_with_shared_word[0]*100./all_ssi_pairs[0], ssi_pairs_with_either_coref_or_word[0]*100./all_ssi_pairs[0])
            # # print 'All Pair statistics:\nShare_coref\tShare_word\tShare_either\n%.2f\t%.2f\t%.2f' \
            # #       % (all_pairs_with_shared_coref[0]*100./all_possible_pairs[0], all_pairs_with_shared_word[0]*100./all_possible_pairs[0], all_pairs_with_either_coref_or_word[0]*100./all_possible_pairs[0])
            #
            # # hist_all_pos = np.histogram(all_pos, bins=max(all_pos)+1)
            # # print 'Histogram of all sent positions: ', util.hist_as_pdf_str(hist_all_pos)
            # # min_sent_len = min(sent_lens)
            # # hist_sent_lens = np.histogram(sent_lens, bins=max(sent_lens)-min_sent_len+1)
            # # print 'min, max sent lens:', min_sent_len, max(sent_lens)
            # # print 'Histogram of sent lens: ', util.hist_as_pdf_str(hist_sent_lens)
            # # min_all_sent_len = min(all_sent_lens)
            # # hist_all_sent_lens = np.histogram(all_sent_lens, bins=max(all_sent_lens)-min_all_sent_len+1)
            # # print 'min, max all sent lens:', min_all_sent_len, max(all_sent_lens)
            # # print 'Histogram of all sent lens: ', util.hist_as_pdf_str(hist_all_sent_lens)
            #
            # # print 'Pearsons r, p value', pearsonr(all_pos, y)
            # # fig, ax1 = plt.subplots(nrows=1)
            # # plt.scatter(all_pos, y)
            # # pp = PdfPages(os.path.join('stuff/plots', FLAGS.dataset_name + '_position_scatter.pdf'))
            # # plt.savefig(pp, format='pdf',bbox_inches='tight')
            # # plt.show()
            # # pp.close()
            #
            # # if FLAGS.dataset_name == 'duc_2004':
            # #     plot_positions(primary_pos_duc, secondary_pos_duc, all_pos_duc)
            #
            # normalized_positions_all = normalized_positions_primary + normalized_positions_secondary
            # # plot_histogram(normalized_positions_primary, num_bins=100)
            # # plot_histogram(normalized_positions_secondary, num_bins=100)
            # # plot_histogram(normalized_positions_all, num_bins=100)
            #
            # sent_lens_together = [sent_lens, all_sent_lens]
            # # plot_histogram(sent_lens_together, pdf=True, start_at_0=True, max_val=70)
            #
            # if FLAGS.dataset_name == 'duc_2004':
            #     distances = distances_duc
            # sent_distances_together = [distances, all_distances]
            # # plot_histogram(sent_distances_together, pdf=True, start_at_0=True, max_val=100)
            #
            # tfidf_similarities_together = [tfidf_similarities, all_tfidf_similarities]
            # # plot_histogram(tfidf_similarities_together, pdf=True, num_bins=100)
            #
            # average_mmrs_together = [average_mmrs, all_average_mmrs]
            # # plot_histogram(average_mmrs_together, pdf=True, num_bins=100)
            #
            # normalized_positions_primary_together = [normalized_positions_primary, bin_values]
            # normalized_positions_secondary_together = [normalized_positions_secondary, bin_values]
            # # plot_histogram(normalized_positions_primary_together, pdf=True, num_bins=100)
            # # plot_histogram(normalized_positions_secondary_together, pdf=True, num_bins=100)
            #
            #
            # list_of_hist_pairs = [
            #     {
            #         'lst': normalized_positions_primary_together,
            #         'pdf': True,
            #         'num_bins': 100,
            #         'y_lim': 3.9,
            #         'y_label': FLAGS.dataset_name,
            #         'x_label': 'Sent position (primary)'
            #     },
            #     {
            #         'lst': normalized_positions_secondary_together,
            #         'pdf': True,
            #         'num_bins': 100,
            #         'y_lim': 3.9,
            #         'x_label': 'Sent position (secondary)'
            #     },
            #     {
            #         'lst': sent_distances_together,
            #         'pdf': True,
            #         'start_at_0': True,
            #         'max_val': 100,
            #         'x_label': 'Sent distance'
            #     },
            #     {
            #         'lst': sent_lens_together,
            #         'pdf': True,
            #         'start_at_0': True,
            #         'max_val': 70,
            #         'x_label': 'Sent length'
            #     },
            #     {
            #         'lst': average_mmrs_together,
            #         'pdf': True,
            #         'num_bins': 100,
            #         'x_label': 'Average TF-IDF importance'
            #     }
            # ]

            print ('Average number of tagged tokens per summary', np.mean(all_num_tagged_tokens))
            print ('Median number of tagged tokens per summary', np.median(all_num_tagged_tokens))
            hist_num_tagged_tokens = np.histogram(all_num_tagged_tokens, bins=max(all_num_tagged_tokens)-min(all_num_tagged_tokens)+1)
            print ('Histogram of number of tagged tokens per summary:', util.hist_as_pdf_str(hist_num_tagged_tokens))

            normalized_positions_pairs_together = [normalized_positions_pairs_first, normalized_positions_pairs_second]
            list_of_hist_pairs = [
                {
                    'lst': [normalized_positions_singles],
                    'pdf': True,
                    'num_bins': 100,
                    # 'y_lim': 3.9,
                    'x_lim': 1.0,
                    'y_label': FLAGS.dataset_name,
                    'x_label': 'Sent Position (Singles)',
                    'legend_labels': ['Primary']
                },
                {
                    'lst': normalized_positions_pairs_together,
                    'pdf': True,
                    'num_bins': 100,
                    # 'y_lim': 3.9,
                    'x_lim': 1.0,
                    'x_label': 'Sent Position (Pairs)',
                    'legend_labels': ['Primary', 'Secondary']
                }
            ]

            all_lists_of_histogram_pairs.append(list_of_hist_pairs)
        with open(plot_data_file, 'wb') as f:
            pickle.dump(all_lists_of_histogram_pairs, f)
    else:
        with open(plot_data_file, 'rb') as f:
            all_lists_of_histogram_pairs = pickle.load(f)
    plot_histograms(all_lists_of_histogram_pairs)

if __name__ == '__main__':
    app.run(main)



