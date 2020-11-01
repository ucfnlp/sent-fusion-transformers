import json
import os
from tqdm import tqdm
from absl import flags
from absl import app
import util
import sys
import glob
import data
import spacy
from spacy.tokens import Doc
import neuralcoref

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'sentence_limit' not in flags.FLAGS:
    flags.DEFINE_integer('sentence_limit', 2, 'Max number of sentences to include for merging.')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')
if 'summinc' not in flags.FLAGS:
    flags.DEFINE_boolean('summinc', False, 'Max number of sentences to include for merging.')
if 'triples_only' not in flags.FLAGS:
    flags.DEFINE_boolean('triples_only', False, 'Max number of sentences to include for merging.')

FLAGS(sys.argv)

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


data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi_and_tag_tokens'
spacy_folder = data_dir + '_spacy'
if FLAGS.summinc:
    spacy_folder += '_summinc'
if FLAGS.triples_only:
    spacy_folder += '_triples'
spacy_data_dir = os.path.join(spacy_folder, FLAGS.dataset_name)
spacy_data_full_dir = os.path.join(spacy_data_dir, 'all')
ssi_dir = 'data/ssi'
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list'), ('article_lcs_paths_list', 'delimited_list_of_list_of_lists')]
# names_to_types = [('raw_article_sents', 'string_list'), ('article', 'string'), ('abstract', 'string_list'), ('doc_indices', 'string')]
# names_to_types = [('raw_article_sents', 'string_list')]
min_matched_tokens = 1
util.create_dirs(spacy_data_full_dir)

def get_sent_and_word_idx_of_mention(mention, article_sent_tokens):
    num_tokens_before = 0
    cur_idx = 0
    for sent_idx, sent in enumerate(article_sent_tokens):
        cur_idx += len(sent)
        if mention.start < cur_idx:
            if mention.text.lower() in util.unfix_bracket_tokens_in_sent(' '.join(sent)):
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

def create_example(writer, groundtruth_summ_sents, raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices, article_lcs_paths_list):
    article_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in raw_article_sents]
    if doc_indices is None:
        doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
    doc_indices = [int(doc_idx) for doc_idx in doc_indices]
    # rel_sent_indices, _, _ = preprocess_for_lambdamart_no_flags.get_rel_sent_indices(doc_indices, article_sent_tokens)
    groundtruth_similar_source_indices_list = util.enforce_sentence_limit(groundtruth_similar_source_indices_list, FLAGS.sentence_limit)
    groundtruth_summ_sent_tokens = [sent.split(' ') for sent in groundtruth_summ_sents[0]]

    a=0
    if FLAGS.summinc:
        all_sents = groundtruth_summ_sents[0] + raw_article_sents
        all_sent_tokens = [util.process_sent(sent, whitespace=True) for sent in all_sents]
    else:
        all_sents = raw_article_sents
        all_sent_tokens = article_sent_tokens
    text = util.unfix_bracket_tokens_in_sent(' '.join(all_sents))
    tokens = text.split(' ')
    sent_start_indices = util.get_sent_start_indices(all_sent_tokens)
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
            mention_dict = {
                'endIndex': end_idx+1,
                'startIndex': start_idx+1,
                'text': mention.text,
                'sentNum': sent_idx+1,
                'isRepresentativeMention': mention == cluster.main,
            }
            mention_list.append(mention_dict)
        cluster_list.append(mention_list)
    new_corefs_str = json.dumps(cluster_list)

    new_tf_example = util.make_tf_example(groundtruth_similar_source_indices_list, raw_article_sents, groundtruth_summary_text,
                                          new_corefs_str, None, article_lcs_paths_list, None, None)
    util.write_tf_example(writer, new_tf_example)

    '''{"endIndex": 20, "type": "NOMINAL", "text": "an investigation into the crash of Germanwings Flight 9525",
     "startIndex": 11, "sentNum": 1, "isRepresentativeMention": false}'''

def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.triples_only and not FLAGS.summinc:
        raise Exception("If using --triples_only, then you must have --summinc on as well.")

    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]

    for dataset_split in dataset_splits:
        source_dir = os.path.join(data_dir, FLAGS.dataset_name)
        source_files = sorted(glob.glob(source_dir + '/' + dataset_split + '*'))

        total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
        example_generator = data.example_generator(source_dir + '/' + dataset_split + '*', True, False,
                                                   should_check_valid=False)

        writer = open(os.path.join(spacy_data_full_dir, dataset_split + '.bin'), 'wb')

        examples_created = 0
        for example_idx, example in enumerate(tqdm(example_generator, total=total)):
            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break
            if examples_created > 6:
                a=0
            _raw_article_sents, _groundtruth_similar_source_indices_list, _groundtruth_summary_text, _corefs, _doc_indices, _article_lcs_paths_list = util.unpack_tf_example(
                example, names_to_types)
            _groundtruth_summ_sents = [[sent.strip() for sent in _groundtruth_summary_text.strip().split('\n')]]
            if FLAGS.triples_only:
                for summ_sent_idx, source_indices in enumerate(_groundtruth_similar_source_indices_list):
                    if len(source_indices) < 2:
                        continue
                    raw_article_sents = [_raw_article_sents[source_indices[0]], _raw_article_sents[source_indices[1]]]
                    groundtruth_summary_text = _groundtruth_summ_sents[0][summ_sent_idx]
                    groundtruth_summ_sents = [[groundtruth_summary_text]]
                    groundtruth_similar_source_indices_list = [source_indices]
                    article_lcs_paths_list = [_article_lcs_paths_list[summ_sent_idx]]
                    create_example(writer, groundtruth_summ_sents, raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, _corefs, None, article_lcs_paths_list)
                    examples_created += 1
            else:
                create_example(writer, _groundtruth_summ_sents, _raw_article_sents, _groundtruth_similar_source_indices_list, _groundtruth_summary_text, _corefs, _doc_indices, _article_lcs_paths_list)
                examples_created += 1


        writer.close()
        if FLAGS.dataset_name == 'cnn_dm' or FLAGS.dataset_name == 'newsroom' or FLAGS.dataset_name == 'xsum':
            chunk_size = 1000
        else:
            chunk_size = 1
        util.chunk_file(dataset_split, spacy_data_full_dir, spacy_data_dir, chunk_size=chunk_size)


if __name__ == '__main__':
    app.run(main)



