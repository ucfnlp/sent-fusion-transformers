import os

import struct

from tensorflow.core.example import example_pb2
from tqdm import tqdm
import pyrouge
from absl import logging
import nltk
import glob
import logging as log
from absl import flags
from absl import app
import shutil
import util
import data
import rouge_functions
import tempfile
tempfile.tempdir = os.path.expanduser('~') + "/tmp"

data_dir = os.path.expanduser('~') + '/data/multidoc_summarization/tf_examples'
log_dir = os.path.expanduser('~') + '/data/discourse/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120


FLAGS = flags.FLAGS


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def get_tf_example(source_file):
    reader = open(source_file, 'rb')
    len_bytes = reader.read(8)
    if not len_bytes: return  # finished reading this file
    str_len = struct.unpack('q', len_bytes)[0]
    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    e = example_pb2.Example.FromString(example_str)
    return e

def get_article_text(source_file):
    e = get_tf_example(source_file)
    article_text = e.features.feature['article'].bytes_list.value[
        0].lower()  # the article text was saved under the key 'article' in the data files
    return article_text

def get_summary_text(summary_file, is_reference):
    with open(summary_file) as f:
        summary_text = f.read()
    return summary_text

def get_human_summary_texts(summary_file):
    summary_texts = []
    e = get_tf_example(summary_file)
    for abstract in e.features.feature['abstract'].bytes_list.value:
        summary_texts.append(abstract)  # the abstracts texts was saved under the key 'abstract' in the data files
    return summary_texts

def split_into_tokens(text):
    tokens = text.split()
    tokens = [t for t in tokens if t != '<s>' and t != '</s>']
    return tokens

def split_into_sent_tokens(text):
    sent_tokens = [[t for t in tokens.strip().split() if t != '<s>' and t != '</s>'] for tokens in text.strip().split('\n')]
    return sent_tokens




def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    source_dir = os.path.join(data_dir, FLAGS.dataset)
    source_files = sorted(glob.glob(source_dir + '/*'))

    for i in range(4):
        ref_dir = os.path.join(log_dir, 'reference_' + str(i), 'reference')
        dec_dir = os.path.join(log_dir, 'reference_' + str(i), 'decoded')
        util.create_dirs(ref_dir)
        util.create_dirs(dec_dir)
        for source_idx, source_file in enumerate(source_files):
            human_summary_texts = get_human_summary_texts(source_file)
            summaries = []
            for summary_text in human_summary_texts:
                summary = data.abstract2sents(summary_text)
                summaries.append(summary)
            candidate = summaries[i]
            references = [summaries[idx] for idx in range(len(summaries)) if idx != i]
            rouge_functions.write_for_rouge(references, candidate, source_idx, ref_dir, dec_dir)

        results_dict = rouge_functions.rouge_eval(ref_dir, dec_dir)
        # print("Results_dict: ", results_dict)
        rouge_functions.rouge_log(results_dict, os.path.join(log_dir, 'reference_' + str(i)))




if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'tac_2011', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    app.run(main)



























