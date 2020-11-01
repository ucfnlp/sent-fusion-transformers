#Import library essentials
from sumy.parsers.html import HtmlParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
import os
from tqdm import tqdm
import pyrouge
from absl import logging
import nltk
import glob
import logging as log
from absl import flags
from absl import app
import shutil
import rouge_functions
import tempfile

import util
import numpy as np

tempfile.tempdir = os.path.expanduser('~') + "/tmp"

summaries_dir = os.path.expanduser('~') + '/data/multidoc_summarization/sumrepo_duc2004'
ref_dir = os.path.expanduser('~') + '/data/multidoc_summarization/sumrepo_duc2004/rouge/reference'
out_dir = os.path.expanduser('~') + '/data/multidoc_summarization/sumrepo_duc2004/rouge'


summary_methods = ['Centroid', 'ICSISumm', 'DPP', 'Submodular']

data_dir = os.path.expanduser('~') + '/data/tf_data'
log_dir = 'logs/'
max_enc_steps = 100
min_dec_steps = 10
max_dec_steps = 30
test_folder = 'decode_test_' + str(max_enc_steps) + \
                        'maxenc_4beam_' + str(min_dec_steps) + 'mindec_' + str(max_dec_steps) + 'maxdec_ckpt-30474'

FLAGS = flags.FLAGS

def extract_digits(text):
    digits = int(''.join([s for s in text if s.isdigit()]))
    return digits

def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.exp_name == 'extractive':
        for summary_method in summary_methods:
            if not os.path.exists(os.path.join(out_dir, summary_method, 'decoded')):
                os.makedirs(os.path.join(out_dir, summary_method, 'decoded'))
            if not os.path.exists(os.path.join(out_dir, summary_method, 'reference')):
                os.makedirs(os.path.join(out_dir, summary_method, 'reference'))
            print((os.path.join(out_dir, summary_method)))
            method_dir = os.path.join(summaries_dir, summary_method)
            file_names = sorted([name for name in os.listdir(method_dir) if name[0] == 'd'])
            for art_idx, article_name in enumerate(tqdm(file_names)):
                file = os.path.join(method_dir, article_name)
                with open(file, 'rb') as f:
                    lines = f.readlines()
                tokenized_sents = [[token.lower() for token in nltk.tokenize.word_tokenize(line)] for line in lines]
                sentences = [' '.join(sent) for sent in tokenized_sents]
                processed_summary = '\n'.join(sentences)
                out_name = '%06d_decoded.txt' % art_idx
                with open(os.path.join(out_dir, summary_method, 'decoded', out_name), 'wb') as f:
                    f.write(processed_summary)

                reference_files = glob.glob(os.path.join(ref_dir, '%06d'%art_idx + '*'))
                abstract_sentences = []
                for ref_file in reference_files:
                    with open(ref_file) as f:
                        lines = f.readlines()
                    abstract_sentences.append(lines)
                rouge_functions.write_for_rouge(abstract_sentences, sentences, art_idx, os.path.join(out_dir, summary_method, 'reference'), os.path.join(out_dir, summary_method, 'decoded'))

            results_dict = rouge_functions.rouge_eval(ref_dir, os.path.join(out_dir, summary_method, 'decoded'))
            # print("Results_dict: ", results_dict)
            rouge_functions.rouge_log(results_dict, os.path.join(out_dir, summary_method))

        for summary_method in summary_methods:
            print(summary_method)
        all_results = ''
        for summary_method in summary_methods:
            sheet_results_file = os.path.join(out_dir, summary_method, "sheets_results.txt")
            with open(sheet_results_file) as f:
                results = f.read()
            all_results += results
        print(all_results)
        a=0

    else:
        # source_dir = os.path.join(data_dir, FLAGS.dataset)
        log_root = os.path.join('logs', FLAGS.exp_name)
        ckpt_folder = util.find_largest_ckpt_folder(log_root)
        print(ckpt_folder)
        # if os.path.exists(os.path.join(log_dir,FLAGS.exp_name,ckpt_folder)):
        if ckpt_folder != 'decoded':
            summary_dir = os.path.join(log_dir,FLAGS.exp_name,ckpt_folder)
        else:
            summary_dir = os.path.join(log_dir,FLAGS.exp_name)
        ref_dir = os.path.join(summary_dir, 'reference')
        dec_dir = os.path.join(summary_dir, 'decoded')
        summary_files = glob.glob(os.path.join(dec_dir, '*_decoded.txt'))
        # summary_files = glob.glob(os.path.join(log_dir + FLAGS.exp_name, 'test_*.txt.result.summary'))
        # if len(summary_files) > 0 and not os.path.exists(dec_dir):      # reformat files from extract + rewrite
        #     os.makedirs(ref_dir)
        #     os.makedirs(dec_dir)
        #     for summary_file in tqdm(summary_files):
        #         ex_index = extract_digits(os.path.basename(summary_file))
        #         new_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)
        #         shutil.copyfile(summary_file, new_file)
        #
        #     ref_files_to_copy = glob.glob(os.path.join(log_dir, FLAGS.dataset, ckpt_folder, 'reference', '*'))
        #     for file in tqdm(ref_files_to_copy):
        #         basename = os.path.basename(file)
        #         shutil.copyfile(file, os.path.join(ref_dir, basename))

        lengths = []
        for summary_file in tqdm(summary_files):
            with open(summary_file) as f:
                summary = f.read()
            length = len(summary.strip().split())
            lengths.append(length)
        print('Average summary length: %.2f' % np.mean(lengths))

        print('Evaluating on %d files' % len(os.listdir(dec_dir)))
        results_dict = rouge_functions.rouge_eval(ref_dir, dec_dir, l_param=FLAGS.l_param)
        rouge_functions.rouge_log(results_dict, summary_dir)


if __name__ == '__main__':
    flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
                        ' If you want to run on human summaries, then enter "reference".')
    flags.DEFINE_string('dataset', 'tac_2011', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    flags.DEFINE_integer('l_param', 100, 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')
    app.run(main)
