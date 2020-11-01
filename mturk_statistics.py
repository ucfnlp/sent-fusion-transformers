import numpy as np
import os
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English

in_folder = os.path.join('mturk','main_task','processed')
in_file = os.path.join(in_folder, 'PoC.tsv')


nlp = English()
try:
    nlp2 = spacy.load('en', disable=['parser', 'ner'])
except:
    nlp2 = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatize_sent(tokens):
    tokens_lemma = [t.lemma_ for t in Doc(nlp.vocab, words=[token for token in tokens])]
    return tokens_lemma

class Pmf(Counter):
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __add__(self, other):
        """Adds two distributions.

        The result is the distribution of sums of values from the
        two distributions.

        other: Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                pmf[key1 + key2] += prob1 * prob2
        return pmf

    def __hash__(self):
        """Returns an integer hash value."""
        return id(self)

    def __eq__(self, other):
        return self is other

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))

with open(in_file, encoding="ISO-8859-1") as f:
# with open(in_file) as f:
    raw_fusions = f.read().strip().split('\n\n')[1:]

fusions = []
for raw_pocs in raw_fusions:
    if raw_pocs == '':
        fusions.append([])
    else:
        pocs = [raw_poc.split('\t') for raw_poc in raw_pocs.split('\n')]
        if pocs[0][-1] == '':
            fusions.append([])
            continue
        pocs = [[int(poc[0]), int(poc[1]), poc[2], poc[3], [int(i) for i in poc[4].split()], 
                poc[5], poc[6], [int(i) for i in poc[7].split()], 
                poc[8], poc[9], [int(i) for i in poc[10].split()], poc[11]] for poc in pocs]
        fusions.append(pocs)

poc_types = []
num_pocs = []
article_ids = []
num_pocs_share_word = []
num_words = []
article_words = []
summ_words = []
for fusion_idx, pocs in enumerate(fusions):
    num_pocs.append(len(pocs))
    for poc in pocs:
        pair_id = poc[0]
        article_id = poc[1]
        sent1 = poc[2]
        sent1_selection = poc[3]
        sent1_selection_indices = poc[4]
        sent2 = poc[5]
        sent2_selection = poc[6]
        sent2_selection_indices = poc[7]
        sent3 = poc[8]
        sent3_selection = poc[9]
        sent3_selection_indices = poc[10]
        poc_type = poc[11]

        poc_types.append(poc_type)
        article_ids.append(article_id)

        share_word = bool(set(lemmatize_sent(sent1_selection.split())) & set(lemmatize_sent(sent2_selection.split())))
        num_pocs_share_word.append(share_word)

        num_words.extend([len(sent1_selection.split()), len(sent2_selection.split()), len(sent3_selection.split())])
        article_words.append(len(sent1.split()) + len(sent2.split()))
        summ_words.append(len(sent3.split()))

poc_type_pmf = Pmf(poc_types)
poc_type_pmf.normalize()
print("Rate of each PoC type", poc_type_pmf)

num_pocs_pmf = Pmf(num_pocs)
num_pocs_pmf.normalize()
print("Portion of fusion instances that had X number of PoCs", num_pocs_pmf)
print("Number of PoCs total: " , len(num_pocs))

print("Number of unique articles:", len(list(set(article_ids))))

print("Percent pocs share at least one word:", np.mean(num_pocs_share_word))

print("Average number of words in each PoC (mean and median):", np.mean(num_words), np.median(num_words))
print("Average number of words in article:", np.mean(article_words))
print("Average number of words in summary:", np.mean(summ_words))


# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
# plt.rc('text', usetex=True)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 2))
fig.tight_layout()
ax1.set_frame_on(False)
ax2.set_frame_on(False)
ax1.axes.get_yaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax1.tick_params(axis='both', which='major', labelsize=7, pad=-3, colors='gray')
ax2.tick_params(axis='both', which='major', labelsize=7, pad=-3, colors='gray')
ax1.xaxis.set_ticks_position('none')
ax2.xaxis.set_ticks_position('none')
# ax1.axis('off')
# ax2.axis('off')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=3)
# Data
r = [0]
raw_data = {'greenBars': [num_pocs_pmf[0]], 'orangeBars': [num_pocs_pmf[1]], 'blueBars': [num_pocs_pmf[2]], 'redBars': [num_pocs_pmf[3]], 'yellowBars': [num_pocs_pmf[4] + num_pocs_pmf[5]]}
df = pd.DataFrame(raw_data)
print(raw_data)

# From raw value to percentage
totals = [i + j + k +l + m for i, j, k, l, m in zip(df['greenBars'], df['orangeBars'], df['blueBars'], df['redBars'], df['yellowBars'])]
greenBars = [i / j * 100 for i, j in zip(df['greenBars'], totals)]
orangeBars = [i / j * 100 for i, j in zip(df['orangeBars'], totals)]
blueBars = [i / j * 100 for i, j in zip(df['blueBars'], totals)]
redBars = [i / j * 100 for i, j in zip(df['redBars'], totals)]
yellowBars = [i / j * 100 for i, j in zip(df['yellowBars'], totals)]

# plot
barWidth = 0.3
names = ('A', 'B', 'C', 'D', 'E')
# Create green Bars
ax1.barh(r, greenBars, color='#edf8e9', edgecolor='black', height=barWidth, label="0")
# Create orange Bars
ax1.barh(r, orangeBars, left=greenBars, color='#bae4b3', edgecolor='black', height=barWidth, label="1")
# Create blue Bars
ax1.barh(r, blueBars, left=[i + j for i, j in zip(greenBars, orangeBars)], color='#74c476', edgecolor='black',
         height=barWidth, label="2")
ax1.barh(r, redBars, left=[i + j + k for i, j, k in zip(greenBars, orangeBars, blueBars)], color='#31a354', edgecolor='black',
         height=barWidth, label="3")
ax1.barh(r, yellowBars, left=[i + j + k + l for i, j, k, l in zip(greenBars, orangeBars, blueBars, redBars)], color='#006d2c', edgecolor='black',
         height=barWidth, label="4+")

# Custom x axis
ax1.set_xticks(r, names)
# ax1.xlabel("group")
ax1.set_title('PoC Counts Per Fusion Instance', {'fontweight' : 'bold'})

# Add a legend
lgd = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -1.6), ncol=5, edgecolor='black')

r = [0]
poc_unique_types = ['Pronoun', 'Name', 'Non-Name', 'Repetition', 'Event']
pretty_types = ['Pronominal', 'Nominal', 'Common-Noun', 'Repetition', 'Event']
raw_data = {'greenBars': [poc_type_pmf[poc_unique_types[0]]], 'orangeBars': [poc_type_pmf[poc_unique_types[1]]], 'blueBars': [poc_type_pmf[poc_unique_types[2]]],
            'redBars': [poc_type_pmf[poc_unique_types[3]]], 'yellowBars': [poc_type_pmf[poc_unique_types[4]]]}
df = pd.DataFrame(raw_data)
print(raw_data)

# From raw value to percentage
totals = [i + j + k +l + m for i, j, k, l, m in zip(df['greenBars'], df['orangeBars'], df['blueBars'], df['redBars'], df['yellowBars'])]
greenBars = [i / j * 100 for i, j in zip(df['greenBars'], totals)]
orangeBars = [i / j * 100 for i, j in zip(df['orangeBars'], totals)]
blueBars = [i / j * 100 for i, j in zip(df['blueBars'], totals)]
redBars = [i / j * 100 for i, j in zip(df['redBars'], totals)]
yellowBars = [i / j * 100 for i, j in zip(df['yellowBars'], totals)]

# plot
barWidth = 0.3
names = ('A', 'B', 'C', 'D', 'E')
# Create green Bars
ax2.barh(r, greenBars, color='#fbb4ae', edgecolor='black', height=barWidth, label="Pronominal")
# Create orange Bars
ax2.barh(r, orangeBars, left=greenBars, color='#b3cde3', edgecolor='black', height=barWidth, label="Nominal")
# Create blue Bars
ax2.barh(r, blueBars, left=[i + j for i, j in zip(greenBars, orangeBars)], color='#ccebc5', edgecolor='black',
         height=barWidth, label="Common-Noun")
ax2.barh(r, redBars, left=[i + j + k for i, j, k in zip(greenBars, orangeBars, blueBars)], color='#decbe4', edgecolor='black',
         height=barWidth, label="Repetition")
ax2.barh(r, yellowBars, left=[i + j + k + l for i, j, k, l in zip(greenBars, orangeBars, blueBars, redBars)], color='#fed9a6', edgecolor='black',
         height=barWidth, label="Event")

# Custom x axis
ax2.set_xticks(r, names)
# ax1.xlabel("group")
# ax2.set_title(r'{\fontsize{30pt}{3em}\selectfont{}{Mean WRFv3.5 LHF\n}{\fontsize{18pt}{3em}\selectfont{}(September 16 - October 30, 2012)}')
ax2.set_title('PoC Type Breakdown', {'fontweight' : 'bold'})

# ax2.text(1.45,-0.08,'a',fontsize=50)
# ax2.text(1.53,-0.08, 'N',fontsize=20)

# Add a legend
lgd2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -2.4), ncol=3, edgecolor='black')



# Show graphic
# plt.show()
plt.savefig(os.path.join('mturk','stats.pdf'), bbox_extra_artists=(lgd,lgd2), bbox_inches='tight')


print('done')