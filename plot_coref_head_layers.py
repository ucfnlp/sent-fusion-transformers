#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from absl import flags
from absl import app
import os
import matplotlib
# if not "DISPLAY" in os.environ:
#     matplotlib.use("Agg")
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
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['xtick.labelsize'] = 14
# plt.rcParams['ytick.labelsize'] = 14
# plt.rcParams['legend.fontsize'] = 14
# plt.rcParams['figure.titlesize'] = 14


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

def plot_positions(results, static_val):
    optional_exp_names = [0,1,2]
    pretty_exp_names = ['1', '2', '≥3']
    pretty_sent_indices = ['CNN/Daily Mail', 'XSum', 'DUC-04']

    ind = np.arange(3)[::-1] * 0.45  # the x locations for the groups
    width = 0.10  # the width of the bars
    interval = np.arange(5) * 0.2

    fig, ax1 = plt.subplots(nrows=1)
    fig.set_size_inches(6, 3.6)
    # plt.subplots_adjust(top=0.7, wspace=0.5, hspace=0.5)

    colors = ['#4078FF', '#2046FF', '#0014FF']
    axes = [ax1]
    hatches = ['//', '', '|']

    exps = [exp_name for exp_name in optional_exp_names]
    ax = axes[0]

    # Data for plotting
    t = np.arange(1, 13)
    s = results

    ax1.plot(t, s, marker='o', label='Trans-ShareRepr')

    t = np.arange(1, 13)
    s = [static_val] * 12
    ax1.plot(t, s, color='black', linestyle='dashed', label='Transformer')

    ax1.set(xlabel='Layer', ylabel='ROUGE-2')
    ax1.set_ylim([19.7,21])
    plt.yticks([20,20.5,21])
    plt.legend(loc="upper right")
    # ax.grid()

    # print rects, optional_exp_names
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_facecolor('#EAEAF2')
    plt.grid(color='white', linewidth=2)
    pp = PdfPages(os.path.join('stuff','coref_head.pdf'))
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    # plt.show()
    pp.close()



def main(unused_argv):
    raw_results = '''20.32
20.02
19.86
19.97
20.68
20.05
20.14
20.41
20.13
20.29
20.13
20.1'''
    res = [float(num) for num in raw_results.split()]
    static_val = 20.03

    plot_positions(res, static_val)
    a=0

if __name__ == '__main__':
    app.run(main)
