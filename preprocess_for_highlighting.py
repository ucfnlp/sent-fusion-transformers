import json
import nltk
import os
from tqdm import tqdm
import numpy as np
from absl import flags
from absl import app
import util
import sys
import glob
import data
from sklearn.feature_extraction.text import CountVectorizer

FLAGS = flags.FLAGS

if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
if 'dataset_split' not in flags.FLAGS:
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset split to use. Must be one of {train, val (or dev), test}')
if 'num_instances' not in flags.FLAGS:
    flags.DEFINE_integer('num_instances', -1,
                         'Number of instances to run for before stopping. Use -1 to run on all instances.')

FLAGS(sys.argv)

data_dir = os.path.expanduser('~') + '/data/tf_data/with_coref_and_ssi'

if FLAGS.dataset_split != 'test':
    correctness_folder = 'correctness_' + FLAGS.dataset_split
else:
    correctness_folder = 'correctness'

ssi_dir = 'data/ssi'
raw_root = 'data/' + correctness_folder + '/raw'
processed_root = 'data/' + correctness_folder + '/processed'
# systems = ['reference', 'novel', 'dca', 'abs-rl-rerank', 'pg', 'bottom-up']
systems = ['reference']
# systems = ['novel']
names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_tuples'), ('summary_text', 'string'), ('corefs', 'json'), ('doc_indices', 'delimited_list')]
min_matched_tokens = 1
preprocess_article_and_human_summaries = True
# nlp = stanfordnlp.Pipeline(lang="en")
# nlp = StanfordNLPLanguage(snlp)


def reorder_list_like(to_reorder, ref_summs, ordered_ref_summs):
    # if len(to_reorder) != len(ref_summs) or len(to_reorder) != len(ordered_ref_summs):
    #     raise Exception('lens of lists are not equal. %d %d %d' % (len(to_reorder), len(ref_summs), len(ordered_ref_summs)))
    print ('Fitting and transforming vecs')
    vec = CountVectorizer(input='content', decode_error='ignore')
    all_vecs = vec.fit_transform(ref_summs + ordered_ref_summs)
    unordered_vecs = all_vecs[:len(ref_summs)]
    ordered_vecs = all_vecs[len(ref_summs):]
    print ('Cosine similarity')
    similarities = util.cosine_similarity(ordered_vecs, unordered_vecs)
    argmaxes = np.argmax(similarities, axis=1)
    indices_found = [False] * len(to_reorder)
    reordered_summaries = []
    for i in tqdm(range(len(argmaxes))):
        argmax_val = argmaxes[i]
        max_val = similarities[i,argmax_val]
        if max_val < 0.7:
            a=0
            # raise Exception('Best result does not match well. \nSystem ref summ: %s\n\n Ordered ref summ: %s' % (ref_summs[argmax_val], ordered_ref_summs[i]))
        # if indices_found[argmax_val]:
        #     raise Exception('Best result was already matched with another ordered ref summ')
        indices_found[argmax_val] = True
        reordered_summaries.append(to_reorder[argmax_val])

    if len(reordered_summaries) != len(to_reorder):
        a=0
        # raise Exception('reordered summaries len (%d) is not equal to original length (%d)' % (len(reordered_summaries), len(to_reorder)))
    return reordered_summaries

def slash_t_to_tab_separated(text):
    fixed_flipped_slash_t_and_newline = text.replace(' <t>\n', '\n<t>')
    new_text = fixed_flipped_slash_t_and_newline.replace(' </t> <t> ', '\t').replace(' </t>\n', '\n').replace('\n<t><t>', '\n').replace('\n<t>', '\n').replace('<t>', '')
    return new_text

def get_sents(text):
    # doc = nlp(text)
    # sents = [' '.join([token.text for token in sent.tokens]) for sent in doc.sentences]
    sents = nltk.tokenize.sent_tokenize(text)
    fixed_sents = []
    for i in range(len(sents)):
        if sents[i].strip() == '.':
            continue
        fixed_sents.append(sents[i])
    new_sents = []
    for i in range(len(fixed_sents))[::-1]:
        if len(fixed_sents[i].strip().split()) <= 2:
            if len(new_sents) == 0:
                if i == 0:
                    new_sents.append(fixed_sents[i])
                else:
                    fixed_sents[i-1] = fixed_sents[i-1] + ' ' + fixed_sents[i]
            else:
                new_sents[0] = fixed_sents[i] + ' ' + new_sents[0]
        else:
            new_sents.insert(0, fixed_sents[i])
    if len(new_sents) != len(sents):
        a=0
    return '\t'.join(new_sents)

def fix_punctuations(text):
    return text.replace(" ''", '"').replace('`` ', '"').replace('` ', "'").replace(" '", "'").replace(' :', ':').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' ,', ',').replace('( ', '(').replace(' )', ')').replace(" 's", "'s").replace(' %', '%').replace('$ ', '$').replace(" 'll", "'ll").replace(" 're", "'re").replace(" n't", "n't")


def main(unused_argv):

    print('Running statistics on %s' % FLAGS.dataset_name)

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    util.create_dirs(processed_root)
    # if not os.path.exists(os.path.join(raw_root, 'reference', 'summaries.txt')):
    util.create_dirs(os.path.join(raw_root, 'reference'))
    util.create_dirs(os.path.join(processed_root, 'article'))
    source_dir = os.path.join(data_dir, FLAGS.dataset_name)
    source_files = sorted(glob.glob(source_dir + '/' + FLAGS.dataset_split + '*'))

    total = len(source_files) * 1000 if ('cnn' in FLAGS.dataset_name or 'newsroom' in FLAGS.dataset_name or 'xsum' in FLAGS.dataset_name) else len(source_files)
    example_generator = data.example_generator(source_dir + '/' + FLAGS.dataset_split + '*', True, False,
                                               should_check_valid=False)

    if preprocess_article_and_human_summaries:
        writer = open(os.path.join(raw_root, 'reference', 'summaries.txt'), 'w')
        writer_article = open(os.path.join(processed_root, 'article', 'articles.txt'), 'w')
        writer_tokenized_article = open(os.path.join(processed_root, 'article', 'articles_tokenized.txt'), 'w')
        reference_articles = []
        for example_idx, example in enumerate(tqdm(example_generator, total=total)):
            if FLAGS.num_instances != -1 and example_idx >= FLAGS.num_instances:
                break
            raw_article_sents, groundtruth_similar_source_indices_list, groundtruth_summary_text, corefs, doc_indices = util.unpack_tf_example(
                example, names_to_types)
            groundtruth_summ_sents = [util.unfix_bracket_tokens_in_sent(sent.strip()) for sent in groundtruth_summary_text.strip().split('\n')]
            writer.write('\t'.join(groundtruth_summ_sents) + '\n')
            reference_article = '\t'.join([util.unfix_bracket_tokens_in_sent(sent.strip()) for sent in raw_article_sents])
            reference_articles.append(reference_article)
            pretty_reference_article = fix_punctuations(reference_article)
            writer_article.write(pretty_reference_article + '\n')
            writer_tokenized_article.write(reference_article + '\n')
        writer.close()

    for system in systems:
        print('Processing ' + system + '...')
        raw_dir = os.path.join(raw_root, system)
        processed_dir = os.path.join(processed_root, system)
        util.create_dirs(processed_dir)
        if system == 'reference':
            with open(os.path.join(raw_dir, 'summaries.txt')) as f:
                with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                    text = f.read()
                    pretty_reference_summaries = fix_punctuations(text)
                    writer.write(pretty_reference_summaries)
                    reference_summaries = [summ.strip() for summ in text.split('\n') if summ.strip() != '']
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    writer_tokenized.write(text + '\n')


        elif system == 'abs-rl-rerank':
            decoded_files = sorted(glob.glob(os.path.join(raw_dir, 'rnn-ext_abs_rl_rerank', 'decoded', '*.dec')))
            sys_ref_files = sorted(glob.glob(os.path.join(raw_dir, 'reference', '*.ref')))
            summaries = []
            for file in decoded_files:
                with open(file) as f:
                    text = f.read()
                    text = util.unfix_bracket_tokens_in_sent(text)
                    summary_sents = text.split('\n')
                    summaries.append('\t'.join(summary_sents))
            sys_ref_summaries = []
            for file in sys_ref_files:
                with open(file) as f:
                    text = f.read()
                    text = util.unfix_bracket_tokens_in_sent(text)
                    summary_sents = text.split('\n')
                    sys_ref_summaries.append('\t'.join(summary_sents))
            reordered_summaries = reorder_list_like(summaries, sys_ref_summaries, reference_summaries)
            with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    for summ in reordered_summaries:
                        writer_tokenized.write(summ + '\n')
                        writer.write(fix_punctuations(summ) + '\n')
        elif system == 'pg':
            decoded_files = sorted(glob.glob(os.path.join(raw_dir, 'pointer-gen-cov', '*_decoded.txt')))
            summaries = []
            for file in tqdm(decoded_files):
                with open(file) as f:
                    summary_sents = f.read().split('\n')
                    summaries.append('\t'.join(summary_sents))
            ref_files = sorted(glob.glob(os.path.join(raw_dir, 'reference', '*_reference.txt')))
            sys_ref_summaries = []
            for file in tqdm(ref_files):
                with open(file) as f:
                    summary_sents = f.read().split('\n')
                    sys_ref_summaries.append('\t'.join(summary_sents))

            reordered_summaries = reorder_list_like(summaries, sys_ref_summaries, reference_summaries)

            with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    for summ in reordered_summaries:
                        writer_tokenized.write(summ + '\n')
                        writer.write(fix_punctuations(summ) + '\n')
        elif system == 'bottom-up':
            with open(os.path.join(raw_dir, 'bottom_up_cnndm_015_threshold.out')) as f:
                text_with_slash_t = f.read()
                text_with_slash_t = util.unfix_bracket_tokens_in_sent(text_with_slash_t)
                text_tab_separated = slash_t_to_tab_separated(text_with_slash_t)
                summaries = [summ.strip() for summ in text_tab_separated.split('\n') if summ.strip() != '']
            with open(os.path.join(raw_dir, 'test.txt.tgt.tagged.shuf.noslash')) as f:
                text_with_slash_t = f.read()
                text_tab_separated = slash_t_to_tab_separated(text_with_slash_t)
                sys_ref_summaries = [summ.strip() for summ in text_tab_separated.split('\n') if summ.strip() != '']
            reordered_summaries = reorder_list_like(summaries, sys_ref_summaries, reference_summaries)
            with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    for summ in reordered_summaries:
                        writer_tokenized.write(summ + '\n')
                        writer.write(fix_punctuations(summ) + '\n')
        elif system == 'dca':
            with open(os.path.join(raw_dir, 'cnndm_m6_m7.txt')) as f:
                text = f.read()
            lines = text.split('\n')
            summary_texts = []
            sys_ref_summary_texts = []
            for line in tqdm(lines[1:]):
                if line.strip() == '':
                    continue
                if len(line.split('\t')) != 3:
                    a=0
                sys_ref_summary, _, summary = line.split('\t')
                summary = summary.replace('u . s .', 'u.s.')
                sys_ref_summary = sys_ref_summary.replace('u . s .', 'u.s.')
                summary_texts.append(summary)
                sys_ref_summary_texts.append(sys_ref_summary)
            summaries = [get_sents(summary) for summary in tqdm(summary_texts)]
            sys_ref_summaries = [get_sents(sys_ref_summary) for sys_ref_summary in tqdm(sys_ref_summary_texts)]
            reordered_summaries = reorder_list_like(summaries, sys_ref_summaries, reference_summaries)
            with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    for summ in reordered_summaries:
                        writer_tokenized.write(summ + '\n')
                        writer.write(fix_punctuations(summ) + '\n')
        elif system == 'novel':
            with open(os.path.join(raw_dir, 'rl-novelty-lm.out')) as f:
                text = f.read()
            lines = text.split('\n')
            summaries = []
            sys_articles = []
            summary_texts = []
            sys_article_texts = []
            for line in tqdm(lines):
                if line.strip() == '':
                    continue
                obj = json.loads(line)
                article = obj['article']
                summary = obj['prediction']
                summary_texts.append(util.unfix_bracket_tokens_in_sent(summary))
                sys_article_texts.append(util.unfix_bracket_tokens_in_sent(article))
            # nlp_summaries = nlp.pipe(summary_texts)
            # nlp_sys_articles = nlp.pipe(sys_article_texts)
            summaries = [get_sents(summary) for summary in tqdm(summary_texts, total=11490)]
            sys_articles = [get_sents(article) for article in tqdm(sys_article_texts, total=11490)]
            reordered_summaries = reorder_list_like(summaries, sys_articles, reference_articles)
            with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    for summ in reordered_summaries:
                        writer_tokenized.write(summ + '\n')
                        writer.write(fix_punctuations(summ) + '\n')
        elif system == 'bert-extr' or system == 'bert-abs':
            decoded_files = sorted(glob.glob(os.path.join(raw_dir, 'decoded', '*_decoded.txt')))
            summaries = []
            for file in tqdm(decoded_files):
                with open(file) as f:
                    summary_sents = f.read().split('\n')
                    summaries.append('\t'.join(summary_sents))
            ref_files = sorted(glob.glob(os.path.join(raw_dir, 'reference', '*_reference.A.txt')))
            sys_ref_summaries = []
            for file in tqdm(ref_files):
                with open(file) as f:
                    summary_sents = f.read().split('\n')
                    sys_ref_summaries.append('\t'.join(summary_sents))

            reordered_summaries = reorder_list_like(summaries, sys_ref_summaries, reference_summaries)

            with open(os.path.join(processed_dir, 'summaries.txt'), 'w') as writer:
                with open(os.path.join(processed_dir, 'summaries_tokenized.txt'), 'w') as writer_tokenized:
                    for summ in reordered_summaries:
                        writer_tokenized.write(summ + '\n')
                        writer.write(fix_punctuations(summ) + '\n')
            a=0









if __name__ == '__main__':
    app.run(main)



