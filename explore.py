import itertools
import os
from tqdm import tqdm
from absl import flags
from absl import app
import util
import sys
import glob
import data
import ssi_functions

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
if 'resolver' not in flags.FLAGS:
    flags.DEFINE_string('resolver', 'spacy', 'Max number of sentences to include for merging.')
if 'save_dataset' not in flags.FLAGS:
    flags.DEFINE_boolean('save_dataset', True, 'Max number of sentences to include for merging.')
if 'summinc' not in flags.FLAGS:
    flags.DEFINE_boolean('summinc', False, 'Max number of sentences to include for merging.')
if 'allcorefs' not in flags.FLAGS:
    flags.DEFINE_boolean('allcorefs', False, 'Max number of sentences to include for merging.')
if 'triples_only' not in flags.FLAGS:
    flags.DEFINE_boolean('triples_only', False, 'Max number of sentences to include for merging.')

FLAGS(sys.argv)


data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens'
if FLAGS.resolver != 'stanford':
    data_dir += '_' + FLAGS.resolver
if FLAGS.summinc:
    data_dir += '_summinc'
if FLAGS.triples_only:
    data_dir += '_triples'
dataset_dir = data_dir
if FLAGS.allcorefs:
    dataset_dir += '_allcorefs'
dataset_dir = os.path.join(dataset_dir + '_fusions', FLAGS.dataset_name)
dataset_full_dir = os.path.join(dataset_dir, 'all')
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]
# names_to_types = [('raw_article_sents', 'string_list'), ('article', 'string'), ('abstract', 'string_list'), ('doc_indices', 'string')]
# names_to_types = [('raw_article_sents', 'string_list')]
min_matched_tokens = 1

out_dir = os.path.join(ssi_dir, 'coref_fusions')
if FLAGS.resolver != 'stanford':
    out_dir += '_' + FLAGS.resolver
if FLAGS.summinc:
    out_dir += '_summinc'
if FLAGS.triples_only:
    out_dir += '_triples'

def get_coref_pairs(corefs):
    coref_pairs = set()
    for coref in corefs:
        sent_indices = set()
        for m in coref:
            sent_idx = m['sentNum'] - 1
            sent_indices.add(sent_idx)
        pairs = list(itertools.combinations(sorted(list(sent_indices)), 2))
        coref_pairs = coref_pairs.union(pairs)
    return list(coref_pairs)

def get_coref_triples_summinc(corefs, num_summ_sents):
    coref_pairs = set()
    for coref in corefs:
        sent_indices = set()
        for m in coref:
            sent_idx = m['sentNum'] - 1
            sent_indices.add(sent_idx)
        triples = list(itertools.combinations(sorted(list(sent_indices)), 3))
        pair_summinc = []
        for triple in triples:
            is_valid_triple = True
            if triple[0] >= num_summ_sents:
                is_valid_triple = False
            if triple[1] < num_summ_sents or triple[2] < num_summ_sents:
                is_valid_triple = False
            if is_valid_triple:
                pair_summinc.append(triple)

        coref_triples = coref_pairs.union(pair_summinc)
    return list(coref_triples)

def get_coref_chain_locations(corefs):
    coref_chain_locations = []
    for coref in corefs:
        my_locs = []
        for m in coref:
            if m['isRepresentativeMention']:
                coref_representative = m['text']
        for m in coref:
            sent_idx = m['sentNum'] - 1
            start_word_idx = m['startIndex'] - 1
            end_word_idx = m['endIndex'] - 1
            my_locs.append((sent_idx, start_word_idx, end_word_idx, coref_representative))
        coref_chain_locations.append(my_locs)
    return coref_chain_locations


def get_coref_fusion_pairs(coref_pairs, groundtruth_similar_source_indices_list):
    coref_fusions = list(set(coref_pairs).intersection(set(groundtruth_similar_source_indices_list)))
    return coref_fusions

def get_coref_fusion_triples(coref_triples, gt_ssi_list):
    shifted_ssi_list = [[source_idx + len(gt_ssi_list) for source_idx in source_indices] for source_indices in gt_ssi_list]
    gt_ssi_list_triples = [tuple([summ_sent_idx] + source_indices) for summ_sent_idx, source_indices in enumerate(shifted_ssi_list)]
    coref_fusions = list(set(coref_triples).intersection(set(gt_ssi_list_triples)))
    return coref_fusions, gt_ssi_list_triples

def get_fusion_locations(coref_chain_locations, groundtruth_similar_source_indices_list):
    new_locations = []
    coref_representatives = []
    consisting_sents_list = []
    for chain in coref_chain_locations:
        pairs = list(itertools.combinations(chain, 2))
        fusion_pairs = [pair for pair in pairs if (pair[0][0], pair[1][0]) in groundtruth_similar_source_indices_list]
        valid_locs = list(set(util.flatten_list_of_lists(fusion_pairs)))
        if len(valid_locs) > 0:
            new_locations.append(valid_locs)
            coref_representatives.append(chain[0][3])
    return new_locations, coref_representatives

def get_fusion_locations_triples(coref_chain_locations, gt_ssi_list_triples):
    new_locations = []
    coref_representatives = []
    for chain in coref_chain_locations:
        triples = list(itertools.combinations(chain, 3))
        fusion_triples = [triple for triple in triples if (triple[0][0], triple[1][0], triple[2][0]) in gt_ssi_list_triples]
        valid_locs = list(set(util.flatten_list_of_lists(fusion_triples)))
        if len(valid_locs) > 0:
            new_locations.append(valid_locs)
            coref_representatives.append(chain[0][3])
    return new_locations, coref_representatives

def filter_only_for_one_fusion(source_indices, fusion_locations, allcorefs=False, triples_only=False):
    new_fusion_locations = []
    for chain in fusion_locations:
        new_chain = []
        first = False
        second = False
        third = False
        for mention in chain:
            if mention[0] == source_indices[0]:
                first = True
                new_mention = list(mention[:3])     # exclude representative mention entry
                new_mention[0] = 0
                new_mention = tuple(new_mention)
                new_chain.append(new_mention)
            elif mention[0] == source_indices[1]:
                second = True
                new_mention = list(mention[:3])     # exclude representative mention entry
                new_mention[0] = 1
                new_mention = tuple(new_mention)
                new_chain.append(new_mention)
            elif triples_only and mention[0] == source_indices[2]:
                third = True
                new_mention = list(mention[:3])     # exclude representative mention entry
                new_mention[0] = 2
                new_mention = tuple(new_mention)
                new_chain.append(new_mention)

        if (allcorefs and len(new_chain) > 0) or (not triples_only and first and second) or (triples_only and first and second and third):
            new_fusion_locations.append(new_chain)
    return new_fusion_locations

def get_summ_sent_idx(source_indices, groundtruth_similar_source_indices_list):
    for summ_sent_idx, cur_source_indices in enumerate(groundtruth_similar_source_indices_list):
        if source_indices == cur_source_indices:
            return summ_sent_idx
    print(source_indices)
    print(groundtruth_similar_source_indices_list)
    raise Exception('Could not find summ sent idx')

def filter_summ_locations(fusion_locations, num_summ_sents):
    summ_locations = [[loc for loc in chain if loc[0] < num_summ_sents] for chain in fusion_locations]
    article_locations = [[loc for loc in chain if loc[0] >= num_summ_sents] for chain in fusion_locations]
    return summ_locations, article_locations



def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]

    for dataset_split in dataset_splits:

        source_dir = os.path.join(data_dir, FLAGS.dataset_name)
        # source_dir = os.path.join(data_dir, FLAGS.dataset_name, 'all')
        source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

        total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
        example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False,
                                                   should_check_valid=False)

        if FLAGS.save_dataset:
            util.create_dirs(dataset_full_dir)
            dataset_writer = open(os.path.join(dataset_full_dir, dataset_split + '.bin'), 'wb')

        num_examples = 0

        coref_fusion_found_idx = 0
        for example_idx, example in enumerate(tqdm(example_generator, total=total)):
            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break
            raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices, article_lcs_paths_list = util.unpack_tf_example(
                example, names_to_types)
            if FLAGS.triples_only:
                groundtruth_similar_source_indices_list = [(0,1)]
            article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
            groundtruth_summ_sents = [[sent.strip() for sent in groundtruth_summary_text.strip().split('\n')]]
            if doc_indices is None:
                doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
            doc_indices = [int(doc_idx) for doc_idx in doc_indices]
            # rel_sent_indices, _, _ = preprocess_for_lambdamart_no_flags.get_rel_sent_indices(doc_indices, article_sent_tokens)
            groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)
            article_lcs_paths_list = util.enforce_sentence_limit(article_lcs_paths_list, FLAGS.sentence_limit)
            groundtruth_similar_source_indices_list, article_lcs_paths_list = util.make_ssi_chronological(groundtruth_similar_source_indices_list, article_lcs_paths_list)
            groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]


            simple_similar_source_indices, lcs_paths_list, _, smooth_article_paths_list = ssi_functions.get_simple_source_indices_list(
                groundtruth_summ_sent_tokens, article_sent_tokens, 3)
            simple_similar_source_indices = util.enforce_sentence_limit(simple_similar_source_indices, 3)
            smooth_article_paths_list = util.enforce_sentence_limit(smooth_article_paths_list, 3)
            simple_similar_source_indices, smooth_article_paths_list = util.make_ssi_chronological(simple_similar_source_indices, smooth_article_paths_list)

            if FLAGS.summinc:
                if len(corefs) == 0:
                    continue
                coref_triples = get_coref_triples_summinc(corefs, len(groundtruth_summ_sents[0]))
                coref_chain_locations = get_coref_chain_locations(corefs)
                coref_fusions, gt_ssi_list_triples = get_coref_fusion_triples(coref_triples, groundtruth_similar_source_indices_list)   # list of source_indices. Each source_indices is a triple (summ_sent_idx, source1_idx, source2idx)
                if len(coref_triples) > 0:
                    for source_indices in gt_ssi_list_triples:
                        if len(source_indices) == 3:
                            a=0
                    a=0
                fusion_locations, coref_representatives = get_fusion_locations_triples(coref_chain_locations, gt_ssi_list_triples)
                summ_fusion_locations, article_fusion_locations = filter_summ_locations(fusion_locations, len(groundtruth_summ_sents[0]))
            else:
                coref_pairs = get_coref_pairs(corefs)
                coref_chain_locations = get_coref_chain_locations(corefs)   # list of chains. Each chain is a list of locs. Each loc is a 4-tuple (sent_idx, start, end, coref_representative)
                coref_fusions = get_coref_fusion_pairs(coref_pairs, groundtruth_similar_source_indices_list)    # list of source_indices. Each source_indices is a pair (source1_idx, source2idx)
                fusion_locations, coref_representatives = get_fusion_locations(coref_chain_locations, groundtruth_similar_source_indices_list)     # list of locs. Each loc is a 4-tuple (sent_idx, start, end, coref_representative)
                article_fusion_locations = fusion_locations
                summ_fusion_locations = None

            if len(coref_fusions) > 0:
                if coref_fusion_found_idx < 200:
                    html = ssi_functions.html_highlight_sents_in_article(groundtruth_summ_sent_tokens, simple_similar_source_indices,
                                                article_sent_tokens, doc_indices=None, lcs_paths_list=lcs_paths_list,
                                                 article_lcs_paths_list=smooth_article_paths_list, fusion_locations=article_fusion_locations, summ_fusion_locations=summ_fusion_locations)
                    ssi_functions.write_highlighted_html(html, out_dir, coref_fusion_found_idx)

                if FLAGS.save_dataset:
                    for source_indices in coref_fusions:
                        my_ssi_list = [source_indices]
                        if FLAGS.triples_only:
                            summ_sent_idx = 0
                        else:
                            summ_sent_idx = get_summ_sent_idx(source_indices, groundtruth_similar_source_indices_list)
                        my_summary_text = groundtruth_summ_sents[0][summ_sent_idx]
                        if FLAGS.allcorefs:
                            my_coref_chains = filter_only_for_one_fusion(source_indices, coref_chain_locations, allcorefs=True, triples_only=FLAGS.triples_only)
                        else:
                            my_coref_chains = filter_only_for_one_fusion(source_indices, fusion_locations, allcorefs=False, triples_only=FLAGS.triples_only)
                        my_article_lcs_paths_list = [article_lcs_paths_list[summ_sent_idx]]
                        new_tf_example = util.make_tf_example(my_ssi_list, raw_article_sents,
                                                              my_summary_text,
                                                              None, None, my_article_lcs_paths_list, my_coref_chains, coref_representatives)
                        util.write_tf_example(dataset_writer, new_tf_example)
                        num_examples += 1

                coref_fusion_found_idx += 1
        print("Num examples created: ", num_examples)

        if FLAGS.save_dataset:
            dataset_writer.close()
            if FLAGS.dataset_name == 'cnn_dm' or FLAGS.dataset_name == 'newsroom' or FLAGS.dataset_name == 'xsum':
                chunk_size = 1000
            else:
                chunk_size = 1
            util.chunk_file(dataset_split, dataset_full_dir, dataset_dir, chunk_size=chunk_size)



if __name__ == '__main__':
    app.run(main)




