import convert_data
from tqdm import tqdm
from absl import app, flags

flags.DEFINE_string('dataset_name', 'cnn_dm', 'Which dataset to use. Makes a log dir based on name.\
                                                Must be one of {tac_2011, tac_2008, duc_2004, duc_tac, cnn_dm} or a custom dataset name')
FLAGS = flags.FLAGS

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)


    gen = convert_data.gigaword_generator(FLAGS.dataset_name, "val")

    arts = []
    abs = []

    for article, abstracts, doc_indices, raw_article_sents in tqdm(gen, total=4000000):
        num_article_tokens = len(raw_article_sents[0].split())
        num_abstract_tokens = len(abstracts[0].split())
        arts.append(num_article_tokens)
        abs.append(num_abstract_tokens)

    import numpy as np

    print(np.histogram(arts, bins=20))
    print(np.histogram(abs, bins=20))

# 70
# 5, 25

'''
gigaword: 70    5, 30
newsroom: 100   5, 50
cnndm_1to1: 40  5, 50
websplit: 60    5, 50

'''


if __name__ == '__main__':
    app.run(main)













