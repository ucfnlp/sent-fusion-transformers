# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""



import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np
from collections import Counter
import json

from best_checkpoint_copier import BestCheckpointCopier
from tqdm import tqdm
import itertools
import beam_search_bert
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", os.path.expanduser('~') + "/models/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'decode', "The name of the task to train.")

flags.DEFINE_string("vocab_file", os.path.expanduser('~') + "/models/uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", os.path.expanduser('~') + "/models/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 5, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 500,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("early_stopping_steps", 500,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("all_steps", None,
                     "How often to save the model checkpoint.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")



flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

if 'singles_and_pairs' not in flags.FLAGS:
    flags.DEFINE_string('singles_and_pairs', 'both', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
if 'dataset_name' not in flags.FLAGS:
    flags.DEFINE_string('dataset_name', 'cnn_dm', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')
flags.DEFINE_string('tfrecords_folder', 'tfrecords', 'Whether to run with only single sentences or with both singles and pairs. Must be in {singles, both}.')

flags.DEFINE_float("mask_prob", 0.7, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("max_predictions_per_seq", 100,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_bool('small_training', False, 'Which dataset split to use. Must be one of {train, val, test}')
flags.DEFINE_integer('input_repeat', 30, 'How many times to repeat the input (this is necessary to change up which tokens in the summary are [MASK]\'ed)')
flags.DEFINE_integer('beam_size', 5, 'Beam size for beam search')
flags.DEFINE_integer('min_dec_steps', 10, 'Beam size for beam search')
flags.DEFINE_integer('max_dec_steps', 40, 'Beam size for beam search')
# flags.DEFINE_boolean('coref', True, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('coref', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('coref_dataset', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_integer('max_chains', 3, 'Beam size for beam search')
flags.DEFINE_integer('coref_head', 4, 'Beam size for beam search')
flags.DEFINE_integer('coref_layer', 4, 'Beam size for beam search')
flags.DEFINE_boolean('link', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('first_chain_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')
flags.DEFINE_boolean('first_mention_only', False, 'If true, save plots of each distribution -- importance, similarity, mmr. This setting makes decoding take much longer.')

np.random.seed(123)


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, sentence_ids=None, article_embedding=None, article_lcs_paths=None, coref_chains=None, sent2_start=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.sentence_ids = sentence_ids
    self.article_embedding = article_embedding
    self.article_lcs_paths = article_lcs_paths
    self.coref_chains = coref_chains
    self.sent2_start = sent2_start


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               positions,
               lm_label_ids,
               label_weights,
               input_sequence_mask,
               # coref_attentions_flattened,
               coref_unique_masks_flattened,
               which_coref_mask_flattened,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.positions = positions
    self.lm_label_ids = lm_label_ids
    self.label_weights = label_weights
    self.input_sequence_mask = input_sequence_mask
    # self.coref_attentions_flattened = coref_attentions_flattened
    self.coref_unique_masks_flattened = coref_unique_masks_flattened
    self.which_coref_mask_flattened = which_coref_mask_flattened
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  def get_token_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_lines(cls, input_file):
    with tf.gfile.Open(input_file, "r") as f:
        lines = f.readlines()
    return lines


class DecodeProcessor(DataProcessor):
  """Processor for the Merge data set."""


  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", None)

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "val.tsv")), "val", None)

  def get_test_examples(self, data_dir):
    if FLAGS.small_training:
        dataset_split = 'train'
    else:
        dataset_split = 'test'
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, dataset_split + ".tsv")), 'test', None)

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def get_delimited_list_of_lists(self, text):
      my_list = text.strip()
      my_list = my_list.split(';')
      return [[int(i) for i in (l.strip().split(' ') if l != '' else [])] for l in my_list]

  def get_delimited_list_of_list_of_lists(self, text):
      my_list = text.strip().split('|')
      return [self.get_delimited_list_of_lists(list_of_lists) for list_of_lists in my_list]

  def remove_nonfirst_mentions(self, chain, sent2_start, summ_sent_start):
      sent1_done = False
      sent2_done = False
      summ_sent_done = False
      new_chain = [None,None]
      for mention in chain:
          try:
              a=mention[0]
          except:
              print(mention)
              print(chain)
              print(sent2_start)
              print(summ_sent_start)
              raise
          if not sent1_done and mention[0] < sent2_start:
              new_chain[0] = mention
              sent1_done = True
          if not sent2_done and mention[0] >= sent2_start and mention[0] < summ_sent_start:
              new_chain[1] = mention
              sent2_done = True
      if new_chain[0] is None or new_chain[1] is None:
          print(chain)
          print(sent2_start)
          print(summ_sent_start)
          print('Could not find mention for both sentences so returning full chain')
          return chain
          # raise Exception('Could not find mention for both sentences')
      return new_chain

  def _create_examples(self, lines, set_type, json_lines):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      # if FLAGS.small_training and i >= 33:
      if (FLAGS.small_training or set_type == 'test') and i >= 33:
          break
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_a = text_a.replace('\xa0','-')
      if line[1] != '':
        text_b = tokenization.convert_to_unicode(line[1])
        text_b = text_b.replace('\xa0','-')
      else:
        text_b = None
      sentence_ids = [0, 1]
      sent2_start = int(line[4])
      if FLAGS.coref or FLAGS.link:
        coref_chains_dict = json.loads(line[5])
        coref_chains = []
        for chain_id in sorted(list(coref_chains_dict.keys())):
            chain_to_add = []
            for mention in coref_chains_dict[chain_id]:
                chain_to_add.append((mention['start'], mention['end']))
            coref_chains.append(chain_to_add)
      else:
        coref_chains = None
      if FLAGS.first_chain_only:
          coref_chains = [coref_chains[0]]
      if FLAGS.first_mention_only:
          summ_sent_start = len(text_a.split(' '))
          coref_chains = [self.remove_nonfirst_mentions(chain, sent2_start, summ_sent_start) for chain in coref_chains]

      num_repeats = 1 if set_type == 'test' else FLAGS.input_repeat
      for _ in range(num_repeats):
        yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None, sentence_ids=sentence_ids, article_embedding=None, article_lcs_paths=None, coref_chains=coref_chains, sent2_start=sent2_start)


def create_token_labels(mappings, article_lcs_paths):
    return [1 if orig_token_idx in article_lcs_paths else 0 for orig_token_idx in mappings]

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def wp_in_first_sent(mappings_a, wp_idx, sent2_start):
    return mappings_a[wp_idx] < sent2_start


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if ex_index == 15180:
      a=0

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        positions=[0] * FLAGS.max_predictions_per_seq,
        lm_label_ids=[0] * FLAGS.max_predictions_per_seq,
        label_weights=[0.0] * FLAGS.max_predictions_per_seq,
        input_sequence_mask=[0] * max_seq_length,
        # coref_attentions_flattened=[0] * (max_seq_length * max_seq_length),
        coref_unique_masks_flattened=[0] * ((FLAGS.max_chains + 1) * max_seq_length),
        which_coref_mask_flattened=[0] * ((FLAGS.max_chains + 1) * max_seq_length),
        is_real_example=False)

  # label_map = {}
  # for (i, label) in enumerate(label_list):
  #   label_map[label] = i
  label_map = tokenizer.vocab
  summ_sent_start = len(example.text_a.split(' '))

  tokens_a, mappings_a = tokenizer.tokenize(example.text_a)
  word_idx_to_wp_indices = {}
  for wp_idx, word_idx in enumerate(mappings_a):
      if word_idx not in word_idx_to_wp_indices:
          word_idx_to_wp_indices[word_idx] = []
      word_idx_to_wp_indices[word_idx].append(wp_idx)
  tokens_b = None
  mappings_b = None
  if example.text_b is not None:
    if FLAGS.do_predict:
        tokens_b = example.text_b.split(' ')
        mappings_b = [0] * len(tokens_b)
    else:
        tokens_b, mappings_b = tokenizer.tokenize(example.text_b)
    word_idx_to_wp_indices_b = {}
    for wp_idx, word_idx in enumerate(mappings_b):
        if word_idx not in word_idx_to_wp_indices_b:
            word_idx_to_wp_indices_b[word_idx] = []
        word_idx_to_wp_indices_b[word_idx].append(wp_idx)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    accounted_max_seq_length = max_seq_length - 3
    if FLAGS.link:
        accounted_max_seq_length -= len(flatten_list_of_lists(example.coref_chains)) * 2
    if len(tokens_a) + len(tokens_b) > accounted_max_seq_length:
        was_truncated = True
    else:
        was_truncated = False
    _truncate_seq_pair(tokens_a, tokens_b, accounted_max_seq_length)
    _truncate_seq_pair(mappings_a, mappings_b, accounted_max_seq_length)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
      mappings_a = mappings_a[0:(max_seq_length - 2)]


  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  mappings = []
  segment_ids = []
  positions = []
  lm_label_tokens = []
  label_weights = []
  wp_idx_to_input_ids_idx = {}
  input_ids_indices_for_poc_tokens = {}
  cur_idx = 0
  num_segment_1 = 0

  # chosen_chain_idx = None
  sent1_mention_done = False
  sent2_mention_done = False

  tokens.append("[CLS]")
  mappings.append(-1)
  segment_ids.append(0)
  wp_idx_to_input_ids_idx[0] = cur_idx
  cur_idx += 1
  num_segment_1 += 1
  for token_idx, token in enumerate(tokens_a):

    if FLAGS.link:
      # Add links for END of POCs
      for chain_idx, chain in enumerate(example.coref_chains):

          if chain_idx >= FLAGS.max_chains:
              continue
          # if chain_idx != 0:
          #     continue
          for mention_idx, mention in enumerate(chain):
              # if mention_idx != 0:
              #     continue
              if mention[1] in word_idx_to_wp_indices:
                  try:
                    end_wp_idx_for_mention = word_idx_to_wp_indices[mention[1]][-1]
                  except:
                    print(mention[1])
                    print(example.coref_chains)
                    print(token)
                    print(tokens_a)
                  if token_idx == end_wp_idx_for_mention:
                    poc_end_token = '[POC-%d-END]' % chain_idx
                    # poc_end_token = ')'
                    # # Randomly convert some of the tokens to the [MASK] token for fine-tuning
                    # if (not FLAGS.do_predict and np.random.rand() < 0.2):
                    #   positions.append(len(tokens))
                    #   lm_label_tokens.append(poc_end_token)
                    #   label_weights.append(1.0)
                    #   tokens.append('[MASK]')
                    # else:
                    tokens.append(poc_end_token)
                    mappings.append(-1)
                    segment_ids.append(0)
                    cur_idx += 1
                    num_segment_1 += 1
      # Add links for START of POCs
      for chain_idx, chain in enumerate(example.coref_chains):
          if chain_idx >= FLAGS.max_chains:
              continue
          # if chain_idx != 0:
          #     continue
          for mention_idx, mention in enumerate(chain):
              # if mention_idx != 0:
              #     continue
              if mention[0] in word_idx_to_wp_indices:
                  try:
                    start_wp_idx_for_mention = word_idx_to_wp_indices[mention[0]][0]
                  except:
                    print(mention[0])
                    print(example.coref_chains)
                    print(token)
                    print(tokens_a)
                  if token_idx == start_wp_idx_for_mention:
                    poc_start_token = '[POC-%d-START]' % chain_idx
                    # poc_start_token = '('
                    # # Randomly convert some of the tokens to the [MASK] token for fine-tuning
                    # if (not FLAGS.do_predict and np.random.rand() < 0.2):
                    #   positions.append(len(tokens))
                    #   lm_label_tokens.append(poc_start_token)
                    #   label_weights.append(1.0)
                    #   tokens.append('[MASK]')
                    # else:
                    tokens.append(poc_start_token)
                    mappings.append(-1)
                    segment_ids.append(0)
                    cur_idx += 1
                    num_segment_1 += 1

    # # Randomly convert some of the tokens to the [MASK] token for fine-tuning
    # if (not FLAGS.do_predict and np.random.rand() < 0.2):
    #   positions.append(len(tokens))
    #   lm_label_tokens.append(token)
    #   label_weights.append(1.0)
    #   tokens.append('[MASK]')
    # else:
    tokens.append(token)
    mappings.append(mappings_a[token_idx])
    segment_ids.append(0)
    wp_idx_to_input_ids_idx[token_idx] = cur_idx
    cur_idx += 1
    num_segment_1 += 1

  tokens.append("[SEP]")
  mappings.append(-1)
  segment_ids.append(0)
  num_segment_1 += 1

  poc_token_counts = Counter([t for t in tokens if 'POC' in t]) + Counter([t for t in lm_label_tokens if 'POC' in t])
  for key in poc_token_counts:
      if 'START' in key:
          other_key = '-'.join(key.split('-')[:-1] + ['END]'])
      else:
          other_key = '-'.join(key.split('-')[:-1] + ['START]'])
      if not was_truncated and poc_token_counts[key] != poc_token_counts[other_key]:
          print(poc_token_counts[key], poc_token_counts[other_key])
          print(poc_token_counts)
          print(tokens, example.coref_chains)
          print(lm_label_tokens)
          raise Exception('POC tokens were added incorrectly.')

  num_segment_2 = 0
  if tokens_b:
    for token_idx, token in enumerate(tokens_b):

        # Randomly convert some of the tokens to the [MASK] token for fine-tuning
        if (FLAGS.do_predict and token_idx == len(tokens_b)-1) or (not FLAGS.do_predict and np.random.rand() < FLAGS.mask_prob):
            positions.append(len(tokens))
            lm_label_tokens.append(token)
            label_weights.append(1.0)
            tokens.append('[MASK]')
        else:
            tokens.append(token)

        mappings.append(mappings_b[token_idx])
        segment_ids.append(1)
        num_segment_2 += 1

    if not FLAGS.do_predict:
        positions.append(len(tokens))
        lm_label_tokens.append("[SEP]")
        label_weights.append(1.0)
        tokens.append('[MASK]')
        mappings.append(-1)
        segment_ids.append(1)
        num_segment_2 += 1

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  if FLAGS.coref:
      chain_wordpiece_indices = []  # List of lists. Each coreference chain has a list of indices representing the wordpieces belonging to it.
      for chain_idx, chain in enumerate(example.coref_chains):
          if chain_idx >= FLAGS.max_chains:
              continue
          my_chain_indices = []
          for mention in chain:
              mention_indices = list(range(mention[0], mention[1]))
              wordpiece_indices = [wp_idx for wp_idx, word_idx in enumerate(mappings_a) if word_idx in mention_indices]
              input_id_indices = [wp_idx_to_input_ids_idx[wp_idx] for wp_idx in wordpiece_indices]
              # wordpiece_indices = flatten_list_of_lists([mappings_a[mention_idx] for mention_idx in mention_indices])
              my_chain_indices.extend(input_id_indices)
          my_chain_indices = sorted(my_chain_indices)
          chain_wordpiece_indices.append(my_chain_indices)
      # coref_attentions = np.ones([max_seq_length, max_seq_length])  # All words that are not part of a coreference chain will have equal attention to all other words (set to 1)
      participating_indices = list(set(flatten_list_of_lists(chain_wordpiece_indices)))
      non_participating_indices = [idx for idx in range(max_seq_length) if idx not in participating_indices]
      # coref_attentions[participating_indices, :] = -10000
      # for row_idx, indices in enumerate(chain_wordpiece_indices):
      #     for idx in indices:
      #       coref_attentions[idx,indices] = 1    # Words that are part of a coreference chain will attend only to other words that are part of its own coreference chain
      # coref_attentions_flattened = coref_attentions.flatten().tolist()

      coref_unique_masks = np.ones([FLAGS.max_chains+1, max_seq_length])
      which_coref_mask = np.zeros([max_seq_length, FLAGS.max_chains+1], dtype=np.int32)
      for row_idx in range(FLAGS.max_chains):
          coref_unique_masks[row_idx, :] = 0
      for row_idx, indices in enumerate(chain_wordpiece_indices):
          coref_unique_masks[row_idx, indices] = 1
          for idx in indices:
            which_coref_mask[idx, row_idx] = 1
      for idx in non_participating_indices:
          which_coref_mask[idx, FLAGS.max_chains] = 1
      which_coref_mask_flattened = which_coref_mask.flatten().tolist()

      for which in which_coref_mask:
          assert np.sum(which) >= 1
          assert np.sum(which) <= FLAGS.max_chains

      coref_unique_masks_flattened = coref_unique_masks.flatten().tolist()
  else:
      coref_attentions = np.zeros([max_seq_length, max_seq_length])
      coref_attentions_flattened = coref_attentions.flatten().tolist()

      coref_unique_masks = np.zeros([FLAGS.max_chains+1, max_seq_length])
      coref_unique_masks_flattened = coref_unique_masks.flatten().tolist()
      which_coref_mask = np.zeros([max_seq_length, FLAGS.max_chains+1], dtype=np.int32)
      which_coref_mask_flattened = which_coref_mask.flatten().tolist()

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  #We won't use these
  token_labels = [0] * len(input_ids)
  sentence_ids = [0] * len(input_ids)
  article_embedding = [0] * (768 * 4)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    token_labels.append(0)
    mappings.append(-2)
    input_mask.append(0)
    segment_ids.append(0)
    sentence_ids.append(0)

  lm_label_ids = tokenizer.convert_tokens_to_ids(lm_label_tokens)
  # Zero-pad up to the max predictions length.
  while len(positions) < FLAGS.max_predictions_per_seq:
    positions.append(0)
    lm_label_ids.append(0)
    label_weights.append(0.0)

  def print_3d(list_of_lists):
      print('\n'.join([" ".join([str(x) for x in row]) for row in list_of_lists]))

  # num_segment_1 = len(tokens_a) + 2
  # num_segment_2 = len(tokens_b) + 1
  input_sequence_mask = []
  for i in range(num_segment_1):
      input_sequence_mask.append(num_segment_1)
  for i in range(num_segment_1, num_segment_1 + num_segment_2):
      input_sequence_mask.append(i+1)
  while len(input_sequence_mask) < max_seq_length:
      input_sequence_mask.append(0)


  assert len(input_ids) == max_seq_length
  assert len(token_labels) == max_seq_length
  assert len(mappings) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(sentence_ids) == max_seq_length
  assert len(positions) == FLAGS.max_predictions_per_seq
  assert len(lm_label_ids) == FLAGS.max_predictions_per_seq
  assert len(label_weights) == FLAGS.max_predictions_per_seq
  assert len(input_sequence_mask) == max_seq_length
  # assert coref_attentions.shape == (max_seq_length, max_seq_length)
  # coref_attentions_str = ''
  # for row in coref_attentions:
  #     for col in row:
  #         coref_attentions_str += str(int(col)) + ' '
  #     coref_attentions_str += '\n'

  label_id = 0
  if ex_index % FLAGS.input_repeat == 0 and (ex_index < FLAGS.input_repeat * 5) and not FLAGS.do_predict:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("positions: %s" % " ".join([str(x) for x in positions]))
    tf.logging.info("lm_label_tokens: %s" % " ".join([str(x) for x in lm_label_tokens]))
    tf.logging.info("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))
    tf.logging.info("label_weights: %s" % " ".join([str(x) for x in label_weights]))
    tf.logging.info("input_sequence_mask: %s" % " ".join([str(x) for x in input_sequence_mask]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    # tf.logging.info("coref_attentions: \n%s" % (coref_attentions_str))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      positions=positions,
      lm_label_ids=lm_label_ids,
      label_weights=label_weights,
      input_sequence_mask=input_sequence_mask,
      # coref_attentions_flattened=coref_attentions_flattened,
      coref_unique_masks_flattened=coref_unique_masks_flattened,
      which_coref_mask_flattened=which_coref_mask_flattened,
      is_real_example=True)
  return feature

def convert_example_to_feature(examples, label_list, max_seq_length, tokenizer, output_file):
    tups = []

    for idx, example in enumerate(examples):
        # if idx > 0:
        #     break
        ex_index = idx
        feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

        features_dict = collections.OrderedDict()

        features_dict["input_ids"] = feature.input_ids
        features_dict["input_mask"] = feature.input_mask
        features_dict["segment_ids"] = feature.segment_ids
        features_dict["label_ids"] = feature.label_id
        features_dict["positions"] = feature.positions
        features_dict["lm_label_ids"] = feature.lm_label_ids
        features_dict["label_weights"] = feature.label_weights
        features_dict["input_sequence_mask"] = feature.input_sequence_mask
        # features_dict["coref_attentions_flattened"] = feature.coref_attentions_flattened
        features_dict["coref_unique_masks_flattened"] = feature.coref_unique_masks_flattened
        features_dict["which_coref_mask_flattened"] = feature.which_coref_mask_flattened


        tup = (features_dict["input_ids"], features_dict["input_mask"], features_dict["segment_ids"], features_dict["label_ids"],
               features_dict["positions"], features_dict["lm_label_ids"], features_dict["label_weights"], features_dict["input_sequence_mask"],
               features_dict["coref_unique_masks_flattened"], features_dict["which_coref_mask_flattened"])
        tups.append(tup)

    return tups


def file_based_convert_examples_to_features(
    example_generator, label_list, max_seq_length, tokenizer, output_file, num_examples):
    # examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  print("Writing examples to .tfrecord file")
  # num_repeats = 1 if 'test' in output_file else FLAGS.input_repeat
  for (ex_index, example) in enumerate(tqdm(example_generator, total=num_examples)):

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["positions"] = create_int_feature(feature.positions)
    features["lm_label_ids"] = create_int_feature(feature.lm_label_ids)
    features["label_weights"] = create_float_feature(feature.label_weights)
    features["input_sequence_mask"] = create_int_feature(feature.input_sequence_mask)
    # features["coref_attentions_flattened"] = create_float_feature(feature.coref_attentions_flattened)
    features["coref_unique_masks_flattened"] = create_float_feature(feature.coref_unique_masks_flattened)
    features["which_coref_mask_flattened"] = create_int_feature(feature.which_coref_mask_flattened)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()



def file_based_input_fn_builder(input_file, seq_length, is_training_or_val,
                                drop_remainder, num_examples=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
      "positions": tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
      "lm_label_ids": tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
      "label_weights": tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
      "input_sequence_mask": tf.FixedLenFeature([seq_length], tf.int64),
      # "coref_attentions_flattened": tf.FixedLenFeature([seq_length*seq_length], tf.float32),
      "coref_unique_masks_flattened": tf.FixedLenFeature([(FLAGS.max_chains+1)*seq_length], tf.float32),
      "which_coref_mask_flattened": tf.FixedLenFeature([(FLAGS.max_chains+1)*seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training_or_val:
      d = d.repeat()
      d = d.shuffle(buffer_size=num_examples)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn



def input_fn_builder(generator):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):

    output_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32,
                    # tf.float32,
                    tf.float32, tf.int32)
    # output_types = tuple([output_types] * 5)
    output_shapes = (tf.TensorShape([FLAGS.max_seq_length]), tf.TensorShape([FLAGS.max_seq_length]), tf.TensorShape([FLAGS.max_seq_length]),
                                                              tf.TensorShape([]), tf.TensorShape([FLAGS.max_predictions_per_seq]), tf.TensorShape([FLAGS.max_predictions_per_seq]),
                                                              tf.TensorShape([FLAGS.max_predictions_per_seq]), tf.TensorShape([FLAGS.max_seq_length]),
                                                            # tf.TensorShape([FLAGS.max_seq_length*FLAGS.max_seq_length]),
                                                            tf.TensorShape([(FLAGS.max_chains+1)*FLAGS.max_seq_length]),
                                                            tf.TensorShape([(FLAGS.max_chains+1)*FLAGS.max_seq_length]))
    # output_shapes = tuple([output_shapes] * 5)
    try:
        dataset = tf.data.Dataset().from_generator(generator, output_types=output_types,
                                                   output_shapes=output_shapes)
    except:
        dataset = tf.data.Dataset.from_generator(generator, output_types=output_types,
                                                   output_shapes=output_shapes)

    iterator = dataset.make_one_shot_iterator()
    features = [iterator.get_next() for _ in range(5)]
    # features = tf.stack(features)
    return {'x': features}

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, positions, lm_label_ids, label_weights, input_mask_2d, coref_attentions):
    """Creates a classification model."""
    if FLAGS.coref:
        coref_layer = FLAGS.coref_layer
        coref_head = FLAGS.coref_head
        # coref_layer = None
        # coref_head = None
        # coref_attentions = None
    else:
        coref_layer = None
        coref_head = None
        coref_attentions = None
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      input_mask_2d=input_mask_2d,
      coref_attentions=coref_attentions,
      coref_layer=coref_layer,
      coref_head=coref_head,
    )

    output_weights = model.get_embedding_table()
    input_tensor = model.get_sequence_output()

    sequence_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]


    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        reshaped_log_probs = tf.reshape(log_probs, [batch_size, FLAGS.max_predictions_per_seq, log_probs.shape[-1]])

        lm_label_ids = tf.reshape(lm_label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            lm_label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, logits, probabilities, log_probs, model.embedding_output, reshaped_log_probs, model.pre_coref_attention_scores, model.attention_scores)
    # return (loss, per_example_loss, log_probs)



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    # tf.logging.info("*** Features ***")
    # for name in sorted(features.keys()):
    #   tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    if mode == tf.estimator.ModeKeys.PREDICT:
      x = features['x']
      input_ids = tf.stack([ex[0] for ex in x])
      input_mask = tf.stack([ex[1] for ex in x])
      segment_ids = tf.stack([ex[2] for ex in x])
      label_ids = tf.stack([ex[3] for ex in x])
      positions = tf.stack([ex[4] for ex in x])
      lm_label_ids = tf.stack([ex[5] for ex in x])
      label_weights = tf.stack([ex[6] for ex in x])
      input_sequence_mask = tf.stack([ex[7] for ex in x])
      # coref_attentions_flattened = tf.stack([ex[8] for ex in x])
      coref_unique_masks_flattened = tf.stack([ex[8] for ex in x])
      which_coref_mask_flattened = tf.stack([ex[9] for ex in x])
      input_mask_2d = tf.sequence_mask(input_sequence_mask, FLAGS.max_seq_length)
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

      sentence_ids, article_embedding, token_labels, mappings = None, None, None, None

    else:
      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      segment_ids = features["segment_ids"]
      label_ids = features["label_ids"]
      positions = features["positions"]
      lm_label_ids = features["lm_label_ids"]
      label_weights = features["label_weights"]
      input_sequence_mask = features["input_sequence_mask"]
      # coref_attentions_flattened = features["coref_attentions_flattened"]
      coref_unique_masks_flattened = features["coref_unique_masks_flattened"]
      which_coref_mask_flattened = features["which_coref_mask_flattened"]
      input_mask_2d = tf.sequence_mask(input_sequence_mask, FLAGS.max_seq_length)
      is_real_example = None
      if "is_real_example" in features:
        is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
      else:
        is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    # coref_attentions_ = tf.reshape(coref_attentions_flattened, [-1, FLAGS.max_seq_length, FLAGS.max_seq_length])

    # coref_indices_list = tf.unstack(tf.reshape(coref_unique_masks_flattened, [-1, FLAGS.max_chains, FLAGS.max_seq_length]))
    coref_unique_masks = tf.cast(tf.reshape(coref_unique_masks_flattened, [-1, FLAGS.max_chains+1, FLAGS.max_seq_length]), tf.int32)
    which_coref_mask = tf.reshape(which_coref_mask_flattened, [-1, FLAGS.max_seq_length, FLAGS.max_chains+1])
    which_coref_mask = tf.transpose(which_coref_mask, [1,0,2])
    which_coref_mask_list = tf.unstack(which_coref_mask)
    coref_masks = []
    for i in range(FLAGS.max_seq_length):
        # # masks = tf.cast(tf.boolean_mask(coref_unique_masks, which_coref_mask_list[i]), tf.bool)
        # masks = tf.broadcast_to(tf.expand_dims(which_coref_mask_list[i], -1), coref_unique_masks.shape)
        # filtered_masks = tf.cast(masks * coref_unique_masks, tf.bool)
        filtered_masks = tf.cast(tf.expand_dims(which_coref_mask_list[i], -1) * coref_unique_masks, tf.bool)
        filtered_masks = tf.transpose(filtered_masks, [1,0,2])
        my_mask = tf.reduce_any(filtered_masks, axis=0)
        coref_masks.append(my_mask)
        if i == 0:
            filtered_masks_ = filtered_masks
            my_mask_ = my_mask
            # mask_ = masks
    coref_attentions = tf.cast(tf.stack(coref_masks), tf.float32)
    coref_attentions = tf.transpose(coref_attentions, [1,0,2])
    pre_input_mask_2d = tf.zeros_like(input_mask_2d)
    if FLAGS.coref:
        coref_attentions = ((coref_attentions - 1) * 10000) + 1




    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, seq_probabilities, log_probs, embedding_output, reshaped_log_probs, pre_coref_attention_scores, attention_scores) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings, positions, lm_label_ids, label_weights, input_mask_2d, coref_attentions)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #   init_string = ""
    #   if var.name in initialized_variable_names:
    #     init_string = ", *INIT_FROM_CKPT*"
    #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                   init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights):
          """Computes the loss and accuracy of the model."""
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
          masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
          masked_lm_accuracy = tf.metrics.accuracy(
              labels=masked_lm_ids,
              predictions=masked_lm_predictions,
              weights=masked_lm_weights)
          masked_lm_mean_loss = tf.metrics.mean(
              values=masked_lm_example_loss, weights=masked_lm_weights)

          print(masked_lm_accuracy)
          print(masked_lm_mean_loss)
          tf.summary.scalar("lm_acc", masked_lm_accuracy[1])
          tf.summary.scalar("lm_loss", masked_lm_mean_loss[1])

          # import nltk
          # nltk.tokenize.toktok
          tf.estimator.EstimatorSpec
          return {
              "masked_lm_accuracy": masked_lm_accuracy,
              "masked_lm_loss": masked_lm_mean_loss
          }

      eval_metrics = (metric_fn, [
          per_example_loss, log_probs, lm_label_ids,
          label_weights
      ])

      tf.summary.scalar("loss", total_loss)

      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=1)
      # logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "coref_attentions": coref_attentions, "-----------------------coref_attentionsoriginal-------------------------": coref_attentions_}, every_n_iter=1)
      # logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "pre_coref_attention_scores": pre_coref_attention_scores, "attention_scores": attention_scores}, every_n_iter=1)
      # logging_hook2 = tf.train.LoggingTensorHook({"pre_coref_attention_scores": pre_coref_attention_scores,}, every_n_iter=1)
      # tf.estimator.EstimatorSpec()
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          eval_metrics=eval_metrics,
          training_hooks=[logging_hook],
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      # def metric_fn(per_example_loss, label_ids, logits, is_real_example, cls_loss, seq_loss):
      #   predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      #   accuracy = tf.metrics.accuracy(
      #       labels=label_ids, predictions=predictions, weights=is_real_example)
      #   loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
      #   _cls_loss = tf.metrics.mean(values=cls_loss, weights=is_real_example)
      #   _seq_loss = tf.metrics.mean(values=seq_loss, weights=is_real_example)
      #   return {
      #       "eval_accuracy": accuracy,
      #       "eval_loss": loss,
      #       "cls_loss": _cls_loss,
      #       "seq_loss": _seq_loss,
      #   }
      #
      # eval_metrics = (metric_fn,
      #                 [per_example_loss, label_ids, logits, is_real_example])
      # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
      #     mode=mode,
      #     loss=total_loss,
      #     eval_metrics=eval_metrics,
      #     scaffold_fn=scaffold_fn)

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights):
          """Computes the loss and accuracy of the model."""
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
          masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
          masked_lm_accuracy = tf.metrics.accuracy(
              labels=masked_lm_ids,
              predictions=masked_lm_predictions,
              weights=masked_lm_weights)
          masked_lm_mean_loss = tf.metrics.mean(
              values=masked_lm_example_loss, weights=masked_lm_weights)

          return {
              "masked_lm_accuracy": masked_lm_accuracy,
              "masked_lm_loss": masked_lm_mean_loss
          }

      eval_metrics = (metric_fn, [
          per_example_loss, log_probs, lm_label_ids,
          label_weights
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      masked_lm_predictions = tf.argmax(
            reshaped_log_probs, axis=-1, output_type=tf.int32)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"masked_lm_predictions": masked_lm_predictions, "input_ids": input_ids, "positions": positions, "lm_label_ids": lm_label_ids,
                       "log_probs": reshaped_log_probs, "coref_attentions": coref_attentions, "coref_unique_masks": coref_unique_masks, "which_coref_mask": tf.transpose(which_coref_mask, [1,0,2]),
                       "filtered_masks": tf.transpose(filtered_masks_, [1,0,2]), "my_mask": my_mask_, "input_mask_2d": input_mask_2d, "pre_input_mask_2d": pre_input_mask_2d},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

def num_lines_in_file(file_path):
    with open(file_path) as f:
        num_lines = sum(1 for line in f)
    return num_lines

"""
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.
    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir), my_input_fn)
    This version supports tf 1.4 and above and can be used by pre-made Estimators like tf.estimator.DNNClassifier. 
    Author: Marc Stogaitis
 """
import tensorflow as tf


class FastPredict:

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        # while not self.closed:
        #     yield self.next_features
        while not self.closed:
            for feature in self.next_features:
                yield feature

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature)
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")

# def main(unused_argv):
#   if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
#     raise Exception("Problem with flags: %s" % unused_argv)

def is_bracket_string(token):
    return token == 'rsb' or token == 'lsb' or token == 'rrb' or token == 'lrb'

def is_contraction_string(token):
    return token == 's' or token == 'll' or token == 're' or token == 'd' or token == 'm' or token == 've'

def create_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def is_number_3_digits_or_less(s):
    return s.isdigit() and len(s) <= 3

def is_long_number(token, token_idx, tokens, consolidated_tokens):
    return token == ',' and len(consolidated_tokens) > 0 and is_number_3_digits_or_less(consolidated_tokens[-1]) and token_idx+1 < len(tokens) and is_number_3_digits_or_less(tokens[token_idx+1])

def is_dashed_word(token, token_idx, tokens, consolidated_tokens):
    return token == '-' and len(consolidated_tokens) > 0 and token_idx+1 < len(tokens) and (not is_bracket_string(tokens[token_idx+1]))

def is_contraction_nt(token, token_idx, tokens, consolidated_tokens):
    return token == "'" and len(consolidated_tokens) > 0 and consolidated_tokens[-1] == 'n' and token_idx+1 < len(tokens) and tokens[token_idx+1] == 't'

def is_bracket(token, token_idx, tokens, consolidated_tokens):
    return is_bracket_string(token) and len(consolidated_tokens) > 0 and consolidated_tokens[-1] == '-' and token_idx+1 < len(tokens) and tokens[token_idx+1] == '-'

def is_decimal(token, token_idx, tokens, consolidated_tokens):
    return token == '.' and len(consolidated_tokens) > 0 and consolidated_tokens[-1].isdigit() and token_idx+1 < len(tokens) and tokens[token_idx+1].isdigit()

def is_dot_com(token, token_idx, tokens, consolidated_tokens):
    return token == '.' and len(consolidated_tokens) > 0 and token_idx+1 < len(tokens) and tokens[token_idx+1] == 'com'

def is_us(token, token_idx, tokens, consolidated_tokens):
    return token == '.' and len(consolidated_tokens) >= 3 and consolidated_tokens[-3] == 'u' and consolidated_tokens[-2] == '.' and consolidated_tokens[-1] == 's'

def is_contraction_nt(token, token_idx, tokens, consolidated_tokens):
    return token == "'" and len(consolidated_tokens) > 0 and consolidated_tokens[-1] == 'n' and token_idx+1 < len(tokens) and tokens[token_idx+1] == 't'

def consolidate_word_pieces(tokens):
    consolidated_tokens = []
    should_skip_next = False
    for token_idx, token in enumerate(tokens):
        if should_skip_next:
            should_skip_next = False
            continue
        if len(token) >= 2 and token[:2] == '##':
            if len(consolidated_tokens) == 0:
                print('Warning: the first token of a sequence started with "##", so adding unchanged to the beginning of sequence.')
            else:
                consolidated_tokens[-1] = consolidated_tokens[-1] + token[2:]
                continue
        if (token == "`" or token == "'") and len(consolidated_tokens) > 0 and consolidated_tokens[-1] == token:
            consolidated_tokens[-1] = consolidated_tokens[-1] + token
        elif is_contraction_string(token) and len(consolidated_tokens) > 0 and consolidated_tokens[-1] == "'":
            consolidated_tokens[-1] = consolidated_tokens[-1] + token
        elif is_long_number(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-1] = consolidated_tokens[-1] + token + tokens[token_idx+1]
            should_skip_next = True
        elif is_dashed_word(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-1] = consolidated_tokens[-1] + token + tokens[token_idx+1]
            should_skip_next = True
        elif is_contraction_nt(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-1] = consolidated_tokens[-1] + token + tokens[token_idx+1]
            should_skip_next = True
        elif is_bracket(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-1] = consolidated_tokens[-1] + token + tokens[token_idx+1]
            should_skip_next = True
        elif is_decimal(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-1] = consolidated_tokens[-1] + token + tokens[token_idx+1]
            should_skip_next = True
        elif is_dot_com(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-1] = consolidated_tokens[-1] + token + tokens[token_idx+1]
            should_skip_next = True
        elif is_us(token, token_idx, tokens, consolidated_tokens):
            consolidated_tokens[-3] = consolidated_tokens[-3] + consolidated_tokens[-2] + consolidated_tokens[-1] + token
            del consolidated_tokens[-2]
            del consolidated_tokens[-1]
        else:
            consolidated_tokens.append(token)
    return consolidated_tokens

def remove_linking_tokens(tokens):
    return [token for token in tokens if 'POC' not in token]

class BertRun:

  def __init__(self):
      return

  '''
  Set-up for train,eval,predict. The code is shared between them. Main() runs mode-specific code.
  '''

  def setUpModel(self):
      tf.logging.set_verbosity(tf.logging.INFO)

      processors = {
          "decode": DecodeProcessor,
      }

      tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                    FLAGS.init_checkpoint)

      data_root = 'data'
      if not os.path.exists(data_root):
          data_root = '../data'
      output_folder = 'output_decoding'
      if FLAGS.coref_dataset or ('poc_dataset' in FLAGS and FLAGS.poc_dataset):
          output_folder += '_crd'
      if FLAGS.coref:
          output_folder += '_coref'
          output_folder += '_l%d_h%d' % (FLAGS.coref_layer, FLAGS.coref_head)
      if FLAGS.link:
          output_folder += '_link'
      if FLAGS.first_chain_only:
          output_folder += '_fc'
      if FLAGS.first_mention_only:
          output_folder += '_fm'
      if FLAGS.small_training:
          output_folder += '_small'
      # output_folder += '_lr%.1E' % FLAGS.learning_rate
      FLAGS.model_dir = os.path.join(data_root, output_folder)
      if FLAGS.do_predict:
          if FLAGS.small_training:
            FLAGS.model_dir = FLAGS.model_dir
          else:
            FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'best')
      FLAGS.data_dir = os.path.join(data_root, FLAGS.dataset_name, FLAGS.singles_and_pairs, 'input_decoding')
      if FLAGS.coref_dataset:
          FLAGS.data_dir += '_crd'
      FLAGS.output_dir = os.path.join(data_root, FLAGS.dataset_name, FLAGS.singles_and_pairs, output_folder)

      if FLAGS.small_training:
          FLAGS.tfrecords_folder += '_small'

      if FLAGS.link:
          FLAGS.vocab_file = "logs/vocab_link.txt"
          if not os.path.exists(FLAGS.vocab_file):
              FLAGS.vocab_file = "../logs/vocab_link.txt"

      print(bcolors.WARNING + "Experiment path: " + FLAGS.output_dir + bcolors.ENDC)

      if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

      bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

      if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

      if FLAGS.all_steps is not None:
          FLAGS.save_checkpoints_steps = FLAGS.all_steps
          FLAGS.iterations_per_loop = FLAGS.all_steps
          FLAGS.early_stopping_steps = FLAGS.all_steps

      tf.gfile.MakeDirs(FLAGS.output_dir)

      task_name = FLAGS.task_name.lower()

      if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

      processor = processors[task_name]()

      label_list = processor.get_labels()

      tokenizer = tokenization.FullTokenizer(
          vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

      tpu_cluster_resolver = None
      if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

      is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
      run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=FLAGS.master,
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          keep_checkpoint_max=3,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=FLAGS.iterations_per_loop,
              num_shards=FLAGS.num_tpu_cores,
              per_host_input_for_training=is_per_host))

      train_example_generator = None
      num_train_examples = None
      num_train_steps = None
      num_warmup_steps = None
      if FLAGS.do_train:
        train_example_generator = processor.get_train_examples(FLAGS.data_dir)
        if FLAGS.small_training:
            num_train_examples = FLAGS.train_batch_size * FLAGS.input_repeat
        else:
            num_train_examples = (num_lines_in_file(os.path.join(FLAGS.data_dir, "train.tsv")) - 1) * FLAGS.input_repeat
        num_train_steps = int(
            num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # train_examples = processor.get_train_examples(FLAGS.data_dir)
        # num_train_steps = int(
        #     len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

      model_fn = model_fn_builder(
          bert_config=bert_config,
          num_labels=len(label_list),
          init_checkpoint=FLAGS.init_checkpoint,
          learning_rate=FLAGS.learning_rate,
          num_train_steps=num_train_steps,
          num_warmup_steps=num_warmup_steps,
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_tpu)

      # If TPU is not available, this will fall back to normal Estimator on CPU
      # or GPU.
      estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size=FLAGS.predict_batch_size)

      self.train_example_generator = train_example_generator
      self.num_train_examples = num_train_examples
      self.label_list = label_list
      self.tokenizer = tokenizer
      self.num_train_steps = num_train_steps
      self.estimator = estimator
      self.processor = processor

  def train(self):
      train_example_generator = self.train_example_generator
      num_train_examples = self.num_train_examples
      num_train_steps = self.num_train_steps
      label_list = self.label_list
      tokenizer = self.tokenizer
      estimator = self.estimator
      processor = self.processor

      early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
          estimator,
          metric_name='loss',
          max_steps_without_decrease=500000,
          min_steps=100,
          run_every_secs=None,
          run_every_steps=FLAGS.early_stopping_steps)

      if FLAGS.do_eval:
        eval_example_generator = processor.get_dev_examples(FLAGS.data_dir)
        if FLAGS.small_training:
            num_actual_eval_examples = FLAGS.train_batch_size * FLAGS.input_repeat
        else:
            num_actual_eval_examples = (num_lines_in_file(os.path.join(FLAGS.data_dir, 'val.tsv')) - 1) * FLAGS.input_repeat
        if FLAGS.use_tpu:
          # TPU requires a fixed batch size for all batches, therefore the number
          # of examples must be a multiple of the batch size, or else examples
          # will get dropped. So we pad with fake examples which are ignored
          # later on. These do NOT count towards the metric (all tf.metrics
          # support a per-instance weight, and these get a weight of 0.0).
          if num_actual_eval_examples % FLAGS.eval_batch_size == 0:
              num_difference = 0
          else:
              num_difference = FLAGS.eval_batch_size - (num_actual_eval_examples % FLAGS.eval_batch_size)
          to_add = [PaddingInputExample() for _ in range(num_difference)]
          eval_example_generator = itertools.chain(eval_example_generator, to_add)
          num_eval_examples_with_padding = num_actual_eval_examples + num_difference
        else:
          num_eval_examples_with_padding = num_actual_eval_examples

        eval_file_name = 'eval_decoding'
        if FLAGS.coref_dataset:
            eval_file_name += '_crd'
        if FLAGS.coref:
            eval_file_name += '_coref'
        if FLAGS.link:
            eval_file_name += '_link'
        if FLAGS.first_chain_only:
            eval_file_name += '_fc'
        if FLAGS.first_mention_only:
            eval_file_name += '_fm'
        eval_file = os.path.join(os.path.dirname(FLAGS.output_dir), FLAGS.tfrecords_folder, eval_file_name + ".tf_record")
        create_dirs(os.path.dirname(eval_file))
        if not os.path.exists(eval_file) or FLAGS.small_training:
            file_based_convert_examples_to_features(
                eval_example_generator, label_list, FLAGS.max_seq_length, tokenizer, eval_file, num_eval_examples_with_padding)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        num_eval_examples_with_padding, num_actual_eval_examples,
                        num_eval_examples_with_padding - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
          assert num_eval_examples_with_padding % FLAGS.eval_batch_size == 0
          eval_steps = int(num_eval_examples_with_padding // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training_or_val=True,
            drop_remainder=eval_drop_remainder,
            num_examples=num_eval_examples_with_padding)

        exporter = BestCheckpointCopier(
            name='best',  # directory within model directory to copy checkpoints to
            checkpoints_to_keep=3,  # number of checkpoints to keep
            score_metric='masked_lm_loss',  # eval_result metric to use to determine "best"
            compare_fn=lambda x, y: x.score < y.score,
            # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
            sort_key_fn=lambda x: x.score,  # key to sort on when discarding excess checkpoints
            sort_reverse=False)  # sort order when discarding excess checkpoints

      if FLAGS.do_train:
        train_file_name = 'train_decoding'
        if FLAGS.coref_dataset:
            train_file_name += '_crd'
        if FLAGS.coref:
            train_file_name += '_coref'
        if FLAGS.link:
            train_file_name += '_link'
        if FLAGS.first_chain_only:
            train_file_name += '_fc'
        if FLAGS.first_mention_only:
            train_file_name += '_fm'
        train_file = os.path.join(os.path.dirname(FLAGS.output_dir), FLAGS.tfrecords_folder, train_file_name + ".tf_record")
        create_dirs(os.path.dirname(train_file))
        if not os.path.exists(train_file) or FLAGS.small_training:
            file_based_convert_examples_to_features(
                train_example_generator, label_list, FLAGS.max_seq_length, tokenizer, train_file, num_train_examples)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", num_train_examples)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training_or_val=True,
            drop_remainder=True,
            num_examples=num_train_examples)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec=tf.estimator.TrainSpec(train_input_fn, hooks=[early_stopping]),
            # eval_spec=tf.estimator.EvalSpec(eval_input_fn, throttle_secs=10)
            eval_spec=tf.estimator.EvalSpec(eval_input_fn, throttle_secs=10, exporters=exporter)
        )

      if FLAGS.do_eval:
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

  def setUpPredict(self):
      train_example_generator = self.train_example_generator
      num_train_examples = self.num_train_examples
      num_train_steps = self.num_train_steps
      label_list = self.label_list
      tokenizer = self.tokenizer
      estimator = self.estimator
      processor = self.processor

      if FLAGS.do_predict:
        dataset_split = 'test'
        predict_name = 'predict'
        if FLAGS.coref_dataset:
            predict_name += '_crd'
        if FLAGS.coref:
            predict_name += '_coref'
        if FLAGS.link:
            predict_name += '_link'
        # predict_example_generator = processor.get_test_examples(FLAGS.data_dir)
        # num_actual_predict_examples = num_lines_in_file(os.path.join(FLAGS.data_dir, dataset_split + '.tsv')) - 1
        # if FLAGS.use_tpu:
        #   # TPU requires a fixed batch size for all batches, therefore the number
        #   # of examples must be a multiple of the batch size, or else examples
        #   # will get dropped. So we pad with fake examples which are ignored
        #   # later on.
        #   if num_actual_predict_examples % FLAGS.eval_batch_size == 0:
        #       num_difference = 0
        #   else:
        #       num_difference = FLAGS.eval_batch_size - (num_actual_predict_examples % FLAGS.eval_batch_size)
        #   to_add = [PaddingInputExample() for _ in range(num_difference)]
        #   num_predict_examples_with_padding = num_actual_predict_examples + num_difference
        # else:
        #   num_predict_examples_with_padding = num_actual_predict_examples

        tf.logging.info("***** Running prediction*****")
        # tf.logging.info("  Num examples = %d (%d actual, %d padding)",
        #                 num_predict_examples_with_padding, num_actual_predict_examples,
        #                 num_predict_examples_with_padding - num_actual_predict_examples)
        # tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_file = os.path.join(os.path.dirname(FLAGS.output_dir), 'tfrecords', predict_name + "_decoding.tf_record")
        create_dirs(os.path.dirname(predict_file))
        print ("Predict file: %s" % predict_file)

        predict_input_fn = input_fn_builder

        classifier = FastPredict(estimator, predict_input_fn)

        def decode_one_step(predict_examples, out_tokens_list):
            for predict_example_idx, predict_example in enumerate(predict_examples):
                out_tokens = out_tokens_list[predict_example_idx]
                predict_example.text_b = ' '.join(out_tokens + ['none'])

            feature_batches = convert_example_to_feature(predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file)
            result = classifier.predict(feature_batches)

            output_predict_file = os.path.join(FLAGS.output_dir, dataset_split + "_results.tsv")
            output_seq_predict_file = os.path.join(FLAGS.output_dir, dataset_split + "_results_seq.tsv")
            output_mappings_predict_file = os.path.join(FLAGS.output_dir, dataset_split + "results_mappings.tsv")

            num_written_lines = 0
            # tf.logging.info("***** Predict results *****")
            topk_ids_list = []
            topk_log_probs_list = []
            input_ids_list = []
            # for (i, prediction) in enumerate(tqdm(result, total=num_actual_predict_examples)):
            for (i, prediction) in enumerate(result):
              masked_lm_predictions = prediction["masked_lm_predictions"]
              input_ids = prediction["input_ids"]
              positions = prediction["positions"]
              lm_label_ids = prediction["lm_label_ids"]
              log_probs = prediction["log_probs"]
              coref_attentions = prediction["coref_attentions"]
              # coref_attentions_ = prediction["coref_attentions_"]
              coref_unique_masks = prediction["coref_unique_masks"]
              which_coref_mask = prediction["which_coref_mask"]
              filtered_masks = prediction["filtered_masks"]
              my_mask = prediction["my_mask"]
              input_mask_2d = prediction["input_mask_2d"]
              pre_input_mask_2d = prediction["pre_input_mask_2d"]
              # mask = prediction["mask"]
              # if i >= num_actual_predict_examples:
              #   break
              # output_line = str(masked_lm_predictions)
              output_line = ' '.join([str(tokenizer.inv_vocab[token_id]) for token_id in input_ids if tokenizer.inv_vocab[token_id] != '[PAD]']) + '\n'
              output_line += ' '.join([str(tokenizer.inv_vocab[token_id]) for token_id in masked_lm_predictions])

              # # printed_line = ''
              # # masked_token_idx = 0
              # # for token_idx in range(len(input_ids)):
              # #     if token_idx in positions and token_idx != 0:
              # #         color = bcolors.WARNING if lm_label_ids[masked_token_idx] == masked_lm_predictions[masked_token_idx] else bcolors.FAIL
              # #         token = color + tokenizer.inv_vocab[masked_lm_predictions[masked_token_idx]] + bcolors.ENDC
              # #         if lm_label_ids[masked_token_idx] != masked_lm_predictions[masked_token_idx]:
              # #             token += '|' + tokenizer.inv_vocab[lm_label_ids[masked_token_idx]]
              # #         masked_token_idx += 1
              # #     else:
              # #         token = tokenizer.inv_vocab[input_ids[token_idx]]
              # #     if token != '[PAD]':
              # #         printed_line += token + ' '
              # # print(printed_line)
              #
              # printed_line = ''
              # masked_token_idx = 0
              # for token_idx in range(len(input_ids)):
              #     token = tokenizer.inv_vocab[input_ids[token_idx]]
              #     if token != '[PAD]':
              #         printed_line += token + ' '
              # print(printed_line)
              #
              # tokens = []
              # for masked_lm_prediction in masked_lm_predictions:
              #     token = tokenizer.inv_vocab[masked_lm_prediction]
              #     tokens.append(token)
              # print(bcolors.WARNING + tokens[0] + bcolors.ENDC)

              final_dist = np.array(log_probs[0])
              topk_ids = final_dist.argsort()[::-1][:FLAGS.beam_size * 2]  # take the k largest probs. note batch_size=beam_size in decode mode
              if 102 in topk_ids:
                  a=0
              topk_log_probs = [final_dist[token_idx] for token_idx in topk_ids]

              topk_ids_list.append(topk_ids)
              topk_log_probs_list.append(topk_log_probs)
              input_ids_list.append(input_ids)

              num_written_lines += 1
            # print (num_written_lines, num_actual_predict_examples)
            # assert num_written_lines == num_actual_predict_examples

            return topk_ids_list, topk_log_probs_list, input_ids_list

        self.decode_one_step = decode_one_step

  def predict(self, text_a, text_b, coref_chains, sent2_start):
        tokenizer = self.tokenizer
        decode_one_step = self.decode_one_step

        predict_example = InputExample(guid=0, text_a=text_a, text_b=text_b, label=0, coref_chains=coref_chains, sent2_start=sent2_start)

        # for ex_idx, predict_example in enumerate(predict_example_generator):
        #     if ex_idx >= 1:
        #         break
        #     out_tokens = []
        #     best_hyp = beam_search_bert.run_beam_search(decode_one_step, tokenizer, predict_example)
        #     print(best_hyp.tokens)

        best_hyp, input_tokens = beam_search_bert.run_beam_search(decode_one_step, tokenizer, predict_example, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
        # print(best_hyp.tokens)
        consolidated_input_tokens = consolidate_word_pieces(input_tokens)
        consolidated_tokens = consolidate_word_pieces(best_hyp.tokens)
        print(' '.join(consolidated_input_tokens))
        print('-----------------------')
        print(' '.join(consolidated_tokens[:-1]))
        print('-----------------------')
        print(text_b)
        print('\n************************************************************\n')
        final_consolidated_tokens = remove_linking_tokens(consolidated_tokens[:-1])
        return final_consolidated_tokens

def main(unused_args):
    bert_run = BertRun()
    bert_run.setUpModel()
    if FLAGS.do_train:
        bert_run.train()
    elif FLAGS.do_predict:
        bert_run.setUpPredict()
        predict_generator = bert_run.processor.get_test_examples(FLAGS.data_dir)
        for predict_example in predict_generator:
            bert_run.predict(predict_example.text_a, predict_example.text_b, predict_example.coref_chains, predict_example.sent2_start)



if __name__ == "__main__":
  # flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  # flags.mark_flag_as_required("output_dir")
  tf.app.run()