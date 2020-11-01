import os
from tqdm import tqdm
from absl import flags
from absl import app
import util
import sys
from collections import Counter
import numpy as np

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')
if 'only_pairs' not in flags.FLAGS:
    flags.DEFINE_boolean('only_pairs', True, 'Run with only_pairs=False first, then only_pairs=True')

FLAGS(sys.argv)

from ssi_functions import get_simple_source_indices_list, html_highlight_sents_in_article
from ssi_functions import write_highlighted_html

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'

ssi_dir = 'data/ssi'
highlight_root = 'data/correctness/highlighted'
processed_root = 'data/correctness/processed'
pairs_only_processed_root = 'data/correctness/processed_only_pairs'
# systems = ['reference']
systems = ['bert-extr', 'bert-abs']
# systems = ['novel', 'dca']
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]
min_matched_tokens = 2
num_summ_sents_per_hit = 6
np.random.seed(123)



def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    stats = {}
    for system in systems:
        print(system)
        summ_sent_lens = []
        processed_dir = os.path.join(processed_root, system)
        outside_article_count = 0
        for example_idx in tqdm(range(11490)):
            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break
            f_summ = open(os.path.join(processed_dir, 'summaries_tokenized.txt'))
            f_article = open(os.path.join(processed_root, 'article', 'articles_tokenized.txt'))
            summary_sent_tokens = [sent.split() for sent in f_summ.readline().strip().split('\t')]
            article_sent_tokens = [sent.split() for sent in f_article.readline().lower().strip().split('\t')]
            summary_tokens = util.flatten_list_of_lists(summary_sent_tokens)
            article_tokens = util.flatten_list_of_lists(article_sent_tokens)
            for summ_token in summary_tokens:
                if summ_token not in article_tokens:
                    outside_article_count += 1
                    break
            summ_sent_lens.extend([len(sent) for sent in summary_sent_tokens])
        stats[system] = np.mean(summ_sent_lens)
        print(outside_article_count)
    print("summ sent len")
    for system in systems:
        print(system)
    for system in systems:
        print("%0.1f" % stats[system])

    util.create_dirs(highlight_root)

    if not FLAGS.only_pairs:
        stats = {}
        for system in systems:
            print('Processing ' + system + '...')
            num_compress = 0
            num_fuse = 0
            num_copy = 0
            num_fail = 0
            highlight_dir = os.path.join(highlight_root, system)
            processed_dir = os.path.join(processed_root, system)
            util.create_dirs(highlight_dir)

            f_ssi = open(os.path.join(processed_dir, 'source_indices.txt'), 'w')
            f_summ = open(os.path.join(processed_dir, 'summaries_tokenized.txt'))
            f_article = open(os.path.join(processed_root, 'article', 'articles_tokenized.txt'))

            for example_idx in tqdm(range(11490)):
                if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                    break
                summary_sent_tokens = [sent.split() for sent in f_summ.readline().strip().split('\t')]
                article_sent_tokens = [sent.split() for sent in f_article.readline().lower().strip().split('\t')]

                groundtruth_ssi_list, lcs_paths_list, article_lcs_paths_list, smooth_article_paths_list = get_simple_source_indices_list(
                    summary_sent_tokens,
                    article_sent_tokens, FLAGS.sentence_limit, min_matched_tokens)
                groundtruth_highlighted_html = html_highlight_sents_in_article(summary_sent_tokens,
                                                                               groundtruth_ssi_list,
                                                                               article_sent_tokens,
                                                                               lcs_paths_list=lcs_paths_list,
                                                                               article_lcs_paths_list=smooth_article_paths_list)
                all_html = '<u>System Summary</u><br><br>' + groundtruth_highlighted_html
                write_highlighted_html(all_html, highlight_dir, example_idx)
                f_ssi.write('\t'.join([','.join(str(idx) for idx in source_indices) if len(source_indices) >= 1 else '-1' for source_indices in groundtruth_ssi_list]) + '\n')
                for ssi_idx, ssi in enumerate(groundtruth_ssi_list):
                    if len(ssi) >= 2:
                        num_fuse += 1
                    elif len(ssi) == 1:
                        source_sent = ' '.join(article_sent_tokens[ssi[0]])
                        summ_sent = ' '.join(summary_sent_tokens[ssi_idx])
                        if source_sent == summ_sent:
                            num_copy += 1
                        else:
                            num_compress += 1
                            # tqdm.write(source_sent + '\n' + summ_sent + '\n\n')
                    else:
                        num_fail += 1
                a=0
            stats[system] = (num_compress, num_fuse, num_copy, num_fail)
            f_summ.close()
            f_article.close()
            f_ssi.close()
        print("num_compress, num_fuse, num_copy, num_fail")
        for system in systems:
            print(system)
            total = sum(stats[system]) * 1.
            print('\t'.join(["%.2f" % (val*100/total) for val in stats[system]]))

    else:
        util.create_dirs(pairs_only_processed_root)
        f_article = open(os.path.join(processed_root, 'article', 'articles.txt'))
        f_summs = []
        f_ssis = []
        all_systems_summs = []
        all_systems_ssis = []
        for sys_idx, system in enumerate(systems):
            processed_dir = os.path.join(processed_root, system)
            f_summ = open(os.path.join(processed_dir, 'summaries.txt'))
            f_ssi = open(os.path.join(processed_dir, 'source_indices.txt'))
            all_systems_summs.append(f_summ.readlines())
            all_systems_ssis.append(f_ssi.readlines())
            # f_summs.append(f_summ)
            # f_ssis.append(f_ssi)


        w_article = open(os.path.join(pairs_only_processed_root, 'article_sents.txt'), 'w')
        w_summ = open(os.path.join(pairs_only_processed_root, 'summaries.txt'), 'w')
        w_ssi = open(os.path.join(pairs_only_processed_root, 'source_indices.txt'), 'w')

        systems_total = []

        for example_idx in tqdm(range(11490)):
            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break

            article_str = f_article.readline()
            article_sents = article_str.split('\t')

            systems_summ_sents = []
            systems_ssis = []
            system_names = []
            no_reference_pairs = False
            ref_summ_sent = None
            ref_source_indices = None
            for sys_idx, system in enumerate(systems):
                system_name = systems[sys_idx]
                # f_summ = f_summs[sys_idx]
                # f_ssi = f_ssis[sys_idx]
                # summary_sents = f_summ.readline().strip().split('\t')
                # ssi = [source_indices_str.split(',') for source_indices_str in f_ssi.readline().strip().split('\t')]
                summary_sents = all_systems_summs[sys_idx][example_idx].strip().split('\t')
                try:
                    ssi = [source_indices_str.split(',') for source_indices_str in all_systems_ssis[sys_idx][example_idx].strip().split('\t')]
                except:
                    a=0
                    raise
                if system_name == 'reference':
                    ssi_pairs = []
                    summary_sents_pairs = []
                    for summ_sent_idx, source_indices in enumerate(ssi):
                        if len(source_indices) == 2:
                            ssi_pairs.append(source_indices)
                            summary_sents_pairs.append(summary_sents[summ_sent_idx])
                    if len(ssi_pairs) == 0:
                        no_reference_pairs = True
                        break
                    summary_sents_pairs, ssi_pairs = util.shuffle(summary_sents_pairs, ssi_pairs)
                    ref_summ_sent = summary_sents_pairs[0]
                    ref_source_indices = ','.join(ssi_pairs[0])
                else:
                    for summ_sent_idx, source_indices in enumerate(ssi):
                        if len(source_indices) == 2:
                            try:
                                systems_summ_sents.append(summary_sents[summ_sent_idx])
                            except:
                                print (len(summary_sents), len(ssi), summ_sent_idx, system, example_idx)
                                raise
                            systems_ssis.append(','.join(ssi[summ_sent_idx]))
                            system_names.append(system_name)
            if no_reference_pairs:
                continue
            if len(systems_summ_sents) < num_summ_sents_per_hit:
                continue
            systems_summ_sents, systems_ssis, system_names = util.shuffle(systems_summ_sents, systems_ssis, system_names)
            systems_summ_sents, systems_ssis, system_names = systems_summ_sents[:num_summ_sents_per_hit-1], systems_ssis[:num_summ_sents_per_hit-1], system_names[:num_summ_sents_per_hit-1]

            systems_summ_sents.append(ref_summ_sent)
            systems_ssis.append(ref_source_indices)
            system_names.append('reference')
            systems_summ_sents, systems_ssis, system_names = util.shuffle(systems_summ_sents, systems_ssis,
                                                                          system_names)

            w_article.write('\t'.join(util.reorder(article_sents, systems_ssis)) + '\n')
            w_summ.write('\t'.join(systems_summ_sents) + '\n')
            w_ssi.write('\t'.join(systems_ssis) + '\n')

            systems_total.extend(system_names)

        print (Counter(systems_total))











if __name__ == '__main__':
    app.run(main)



