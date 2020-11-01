from sklearn.feature_extraction.text import TfidfVectorizer
import time
import numpy as np
import data
from tqdm import tqdm
import util
from absl import flags
from absl import app
import os
import struct
import glob
from tensorflow.core.example import example_pb2
from scoop import futures
import pickle
import sys
import spacy
import html.parser
from nltk.stem.porter import PorterStemmer
import dill

FLAGS = flags.FLAGS

flags.DEFINE_boolean('pca', False,
                     'If true, save plots of each distribution -- im')
flags.DEFINE_string('input_dataset', 'all',
                     'If true, save plots of each distribution -- im')
flags.DEFINE_boolean('pg_mmr', False,
                     'If true, save plots of each distribution -- im')


FLAGS(sys.argv)

dataset_split = 'all'
num_instances = -1,
random_seed = 123
max_sent_len_feat = 20
balance = True
importance = True
real_values = True
singles_and_pairs = 'singles'
include_sents_dist = True
lr = False
min_df = 5

data_dir = os.path.expanduser('~') + '/data/tf_data'
log_dir = 'logs'
out_dir = 'data/tfidf'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

names_to_types = [('article', 'string')]


def get_tf_example(source_file):
    reader = open(source_file, 'rb')
    len_bytes = reader.read(8)
    if not len_bytes: return  # finished reading this file
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    e = example_pb2.Example.FromString(example_str)
    return e

def get_summary_text(summary_file):
    with open(summary_file) as f:
        summary_text = f.read()
    return summary_text

def get_summary_from_example(e):
    summary_texts = []
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract)  # the abstracts texts was saved under the key 'abstract' in the data files
    all_abstract_sentences = [[sent.strip() for sent in data.abstract2sents(
        abstract)] for abstract in summary_texts]
    summary_text = '\n'.join(all_abstract_sentences[0])
    return summary_text


def get_human_summary_texts(summary_file):
    summary_texts = []
    e = get_tf_example(summary_file)
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract)  # the abstracts texts was saved under the key 'abstract' in the data files
    all_abstract_sentences = [[sent.strip() for sent in data.abstract2sents(
        abstract)] for abstract in summary_texts]
    summary_text = '\n'.join(all_abstract_sentences[0])
    return summary_text

def split_into_tokens(text):
    tokens = text.split()
    tokens = [t for t in tokens if t != '<s>' and t != '</s>']
    return tokens

def split_into_sent_tokens(text):
    sent_tokens = [[t for t in tokens.strip().split() if t != '<s>' and t != '</s>'] for tokens in text.strip().split('\n')]
    return sent_tokens




def save_as_txt_file(ex):
    # example_idx += 1
    # if num_instances != -1 and example_idx >= num_instances:
    #     break
    example, example_idx = ex
    # print example_idx
    article_text, = util.unpack_tf_example(example, names_to_types)
    article_text = article_text
    # out_path = os.path.join(out_dir, in_dataset, 'article_%06d.txt' % example_idx)
    # with open(out_path, 'wb') as f:
    #     f.write(article_text)
    return article_text



def example_generator_extended(example_generator, total):
    example_idx = -1
    for example in tqdm(example_generator, total=total):
    # for example in example_generator:
        example_idx += 1
        if num_instances != -1 and example_idx >= num_instances:
            break
        yield (example, example_idx)



# create a spaCy tokenizer
spacy.load('en')
lemmatizer = spacy.lang.en.English()
html_parser = html.parser.HTMLParser()

# remove html entities from docs and
# set everything to lowercase
def my_preprocessor(doc):
    return(html_parser.unescape(doc).lower())

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])


def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    start_time = time.time()
    np.random.seed(random_seed)
    util.create_dirs(os.path.join(out_dir, FLAGS.input_dataset))

    if FLAGS.input_dataset == 'all':
        datasets = ['duc_2003', 'duc_2004', 'tac_2008', 'tac_2010', 'tac_2011', 'cnn_dm', 'xsum']
    else:
        datasets = [FLAGS.input_dataset]
    if dataset_split == 'all':
        dataset_splits = ['train', 'val', 'test']
    else:
        dataset_splits = [dataset_split]
    all_articles = []
    for in_dataset in datasets:
        source_dir = os.path.join(data_dir, in_dataset)

        for split in dataset_splits:
            # split = dataset_split
            source_files = sorted(glob.glob(source_dir + '/' + split + '*'))

            if len(source_files) == 0:
                continue

            total = len(source_files)*1000 if 'cnn' or 'newsroom' in in_dataset else len(source_files)
            example_generator = data.example_generator(source_dir + '/' + split + '*', True, False, should_check_valid=False)
            # for example in tqdm(example_generator, total=total):
            ex_gen = example_generator_extended(example_generator, total)
            print('Creating list')
            ex_list = [ex for ex in ex_gen]
            print('Converting...')

            articles = list(futures.map(save_as_txt_file, ex_list))
            all_articles.extend(articles)
    vec = TfidfVectorizer(input='content', ngram_range=(1,1), min_df=min_df, max_df=0.5, decode_error='ignore', preprocessor=my_preprocessor, tokenizer=my_tokenizer)

    # list(futures.map(save_as_txt_file, ex_list))
    # file_list = [os.path.join(out_dir, in_dataset, fname) for fname in os.listdir(os.path.join(out_dir, in_dataset))]
    # vec = TfidfVectorizer(input='filename', ngram_range=(1,1), min_df=min_df, max_df=0.5, decode_error='ignore')
    # vec.fit(file_list)

    if FLAGS.pca:
        X = vec.fit_transform(all_articles)
        suffix = '_pca'
    elif FLAGS.pg_mmr:
        stemmer = PorterStemmer()

        class StemmedTfidfVectorizer(TfidfVectorizer):
            def build_analyzer(self):
                analyzer = super(TfidfVectorizer, self).build_analyzer()
                return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

        vec = StemmedTfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), max_df=0.7)
        vec.fit_transform(all_articles)
    else:
        vec.fit_transform(all_articles)
        suffix = ''
    print('Vocabulary size', len(list(vec.vocabulary_.keys())))
    if FLAGS.pg_mmr:
        util.create_dirs(os.path.join(log_dir, 'tfidf_vectorizer'))
        with open(os.path.join(log_dir, 'tfidf_vectorizer', FLAGS.input_dataset + '.dill'), 'wb') as f:
            dill.dump(vec, f)
    else:
        with open(os.path.join(out_dir, FLAGS.input_dataset + '_tfidf_vec_' + str(min_df) + suffix + '.pkl'), 'wb') as f:
            pickle.dump(vec, f)

    if FLAGS.pca:
        print('Fitting LSA model...')
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=100)
        svd.fit(X)
        with open(os.path.join(out_dir, FLAGS.input_dataset + '_pca' + '.pkl'), 'wb') as f:
            pickle.dump(svd, f)




    util.print_execution_time(start_time)


if __name__ == '__main__':

    app.run(main)






























