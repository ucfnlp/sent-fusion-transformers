#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from absl import flags
from absl import app
import os
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['hatch.linewidth'] = 1.0  # previous pdf hatch linewidth
matplotlib.rcParams['hatch.linewidth'] = 1.0  # previous svg hatch linewidth

# font = {'family' : 'serif',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)

plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 14


FLAGS = flags.FLAGS

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

data_dir = os.path.expanduser('~') + '/data/multidoc_summarization/tf_examples'
log_dir = os.path.expanduser('~') + '/data/multidoc_summarization/logs/'
max_enc_steps = 100000
min_dec_steps = 100
max_dec_steps = 120

# flags.DEFINE_string('exp_name', 'reference', 'Path to system-generated summaries that we want to evaluate.' +
#                            ' If you want to run on human summaries, then enter "reference".')
# flags.DEFINE_string('dataset', 'tac_2011', 'Which dataset to use. Can be {duc_2004, tac_2011, etc}')



def label_barh(ax, bars, texts, is_inside=True, **kwargs):
    """
    Attach a text label to each horizontal bar displaying its y value
    """
    max_y_value = max(bar.get_height() for bar in bars)
    if is_inside:
        distance = max_y_value * 1
    else:
        distance = max_y_value * 7


    for bar_idx, bar in enumerate(bars):
        text = texts[bar_idx]
        if is_inside:
            text_x = bar.get_width() - distance
        else:
            text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)

def plot_positions(results):
    optional_exp_names = [0,1,2]
    pretty_exp_names = ['1'.decode('utf-8'), '2'.decode('utf-8'), '≥3'.decode('utf-8')]
    pretty_sent_indices = ['CNN/Daily Mail', 'XSum', 'DUC-04']

    ind = np.arange(3)[::-1] * 0.45  # the x locations for the groups
    width = 0.10  # the width of the bars
    interval = np.arange(5) * 0.2

    fig, ax1 = plt.subplots(nrows=1)
    fig.set_size_inches(6, 4)
    plt.subplots_adjust(top=0.7, wspace=0.5, hspace=0.5)

    colors = ['#4078FF', '#2046FF', '#0014FF']
    axes = [ax1]
    hatches = ['//', '', '|']

    exps = [exp_name for exp_name in optional_exp_names]
    ax = axes[0]
    rects = []
    for sent_idx in range(3):
        medians = [results[exp,sent_idx] for exp in exps]
        # ci = [[val[1] - val[0] for val in values], [val[2] - val[1] for val in values]]
        hatch = hatches[sent_idx]
        # rect = ax.barh(ind - width * sent_idx, medians, height=width, color=colors[sent_idx],
        #                alpha=0.8, edgecolor='black', hatch=None, capsize=2.5)
        rect = ax.barh(ind - width * sent_idx, medians, height=width,
                       alpha=0.8, edgecolor='black', hatch=hatch, capsize=2.5)
        rects.append(rect)

    ax1.set_xlabel('Summary sentences (%)')
    ax1.set_yticks(ind - width * 1)
    ax1.set_yticklabels(pretty_exp_names, rotation='vertical', fontdict={'verticalalignment': 'center'})
    # print rects, optional_exp_names
    ax1.legend([r[0] for r in rects],
               pretty_sent_indices, prop={'size': 11}, ncol=1,
               edgecolor='black', shadow=False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    pp = PdfPages('sent_positions.pdf')
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    plt.show()
    pp.close()



def main(unused_argv):
    raw_results = '''3.63	63.51	29.49	3.12	0.23	0.02
12.5	39.18	34.81	13.51	0	0
6.22	45.41	33.51	11.9	2.27	0.68'''
    res = [[float(num) for num in line.split()] for line in raw_results.split('\n')]
    res = np.array(res)
    col1 = res[:,1]
    col2 = res[:,2]
    col3 = np.sum(res[:,3:], axis=1)
    results = np.stack([col1, col2, col3])

    plot_positions(results)
    a=0

if __name__ == '__main__':
    app.run(main)





# def plot_positions(results):
#
#     datasets = ['duc_2004', 'tac_2011']
#     optional_exp_names = ['_original', '_original_randomize_sent_order', '_reservoir_lambda_0.6_mute_7_tfidf']
#     pretty_exp_names = ['PG-Original', 'PG-Original (Rand Input)', 'PG-MMR']
#
#     ind = np.arange(5)  # the x locations for the groups
#     width = 0.2  # the width of the bars
#
#     fig, (ax1, ax2) = plt.subplots(nrows=2)
#     plt.subplots_adjust(bottom=0.4, wspace=0.5)
#
#     colors = ['#9E9E9E', '#BDBDBD', '#757575', '#424242']
#     axes = [ax1, ax2]
#     hatches = ['//', '\\\\', '||']
#
#     for dataset_idx, dataset in enumerate(datasets):
#         exps = [dataset + exp_name for exp_name in optional_exp_names]
#         ax = axes[dataset_idx]
#         rects = []
#         for exp_idx, exp in enumerate(exps):
#             values = results[exp]
#             medians = [val[1] for val in values]
#             ci = [[val[1] - val[0] for val in values], [val[2] - val[1] for val in values]]
#             hatch = hatches[exp_idx]
#             rect = ax.bar(ind + width*exp_idx, medians, yerr=ci, width=width, alpha=0.8, edgecolor='black', hatch=hatch, capsize=2.5)
#             rects.append(rect)
#
#
#     # ax.set_title('effect of a coverage-based regularizer and beam search with reference')
#     ax1.set_ylabel('DUC-04', fontsize=12)
#     ax2.set_ylabel('TAC-11', fontsize=12)
#     ax1.set_xticks(ind + width * 4 / 2)
#     ax2.set_xticks(ind + width * 4 / 2)
#     ax1.set_xticklabels(('', '', '', '', ''), fontsize=1)
#     ax2.set_xticklabels(('1st', '2nd', '3rd', '4th', '5th'), fontsize=12)
#     print rects, optional_exp_names
#     ax2.legend([r[0] for r in rects],
#                pretty_exp_names, prop={'size': 11}, ncol=2,
#                edgecolor='black', loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=False)
#
#     ax1.set_ylim([0, 105])
#     ax2.set_ylim([0, 105])
#
#     pp = PdfPages('sent_positions.pdf')
#     plt.savefig(pp, format='pdf')
#     pp.close()
