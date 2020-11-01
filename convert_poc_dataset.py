
import struct
from tensorflow.core.example import example_pb2
import os
from absl import flags
from absl import app
from tqdm import tqdm
import util
import sys

# try:
#     reload(sys)
#     sys.setdefaultencoding('utf8')
# except:
#     a = 0

FLAGS = flags.FLAGS

in_path = 'mturk/main_task/processed/PoC.tsv'
out_data_path = os.path.expanduser('~') + '/data/tf_data/poc_fusions'

def process_dataset():
    out_dir = os.path.join(out_data_path, FLAGS.dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    gen = poc_generator()
    num_examples = 1599
    write_with_generator(gen, num_examples, out_dir, FLAGS.dataset_split)

def get_coref_chains(pocs):
    coref_chains = [[[0] + poc[4], [1] + poc[7]] for poc in pocs]
    return coref_chains

def coref_chains_to_string(coref_chains):
    return '|'.join([';'.join([' '.join([str(i) for i in mention]) for mention in chain]) for chain in coref_chains])

def poc_generator():
    with open(in_path, encoding = "ISO-8859-1") as f:
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
    for pocs in fusions:
        if len(pocs) == 0:
            continue
        coref_chains = get_coref_chains(pocs)
        coref_chains_str = coref_chains_to_string(coref_chains)
        poc = pocs[0]
        raw_article_sents = [process_article(poc[2]), process_article(poc[5])]
        summary_sent = process_abstract(poc[8])
        tf_example = make_example(summary_sent, raw_article_sents, coref_chains_str)
        yield tf_example



def process_article(abstract):
    abstract = abstract.replace('\x92', "'")
    tokenized_sent = abstract.split()
    tokenized_sent = [util.fix_bracket_token(token) for token in tokenized_sent]
    abstract = ' '.join(tokenized_sent)
    abstract = abstract.strip()
    return abstract

def process_abstract(abstract):
    abstract = abstract.lower()
    abstract = abstract.replace('\x92', "'")
    tokenized_sent = abstract.split()
    tokenized_sent = [util.fix_bracket_token(token) for token in tokenized_sent]
    abstract = ' '.join(tokenized_sent)
    abstract = abstract.strip()
    return abstract


def make_example(summary_text, raw_article_sents, coref_chains_str):
    tf_example = example_pb2.Example()
    tf_example.features.feature['summary_text'].bytes_list.value.extend([util.encode_text(summary_text)])
    if raw_article_sents is not None:
        for sent in raw_article_sents:
            tf_example.features.feature['raw_article_sents'].bytes_list.value.extend([util.encode_text(sent)])
    tf_example.features.feature['coref_chains'].bytes_list.value.extend([util.encode_text(coref_chains_str)])
    return tf_example


def write_tf_example(tf_example, writer):
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

def write_with_generator(example_generator, num_examples, out_dir, data_split):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_idx = 1
    out_file_name = os.path.join(out_dir, data_split + '_{:05d}.bin'.format(out_idx // 1000 + 1))
    writer = open(os.path.join(out_file_name), 'wb')
    for example in tqdm(example_generator, total=num_examples):
        if (out_idx - 1) % 1000 == 0 and out_idx != 1:
            writer.close()
            out_file_name = os.path.join(out_dir, data_split + '_{:05d}.bin'.format(out_idx // 1000 + 1))
            writer = open(os.path.join(out_file_name), 'wb')
        write_tf_example(example, writer)

        out_idx += 1
    writer.close()
    a = 0


def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    if FLAGS.dataset_name == '':
        raise Exception('Must specify which dataset to convert.')
    process_dataset()


if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'cnn_dm',
                        'Which dataset to convert from raw data to tf examples')
    flags.DEFINE_string('dataset_split', 'test', 'Which dataset to convert from raw data to tf examples')
    app.run(main)

























