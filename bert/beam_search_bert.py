# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications made 2020 by Logan Lebanoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding"""
import copy


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, input_example, already_added):
        """Hypothesis constructor.

        Args:
            tokens: List of integers. The ids of the tokens that form the summary so far.
            log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
            state: Current state of the decoder, a LSTMStateTuple.
            attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
            p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
            coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.input_example = input_example
        self.already_added = already_added

    def extend(self, token, log_prob):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
            token: Integer. Latest token produced by beam search.
            log_prob: Float. Log prob of the latest token.
            state: Current decoder state, a LSTMStateTuple.
            attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
            p_gen: Generation probability on latest step. Float.
            coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
            New Hypothesis for next step.
        """
        if self.does_trigram_exist(token):
            log_prob = -1000
        input_example = copy.deepcopy(self.input_example)
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          input_example=input_example,
                          already_added=self.already_added)

    def does_trigram_exist(self, token):
        if len(self.tokens) < 2:
            return False
        candidate_trigram = self.tokens[-2:] + [token]
        for i in range(len(self.tokens)-2):
            if self.tokens[i:i+3] == candidate_trigram:
                return True
        return False

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)

def results_still_too_small(results, beam_size):
    return len(results) < beam_size

# We have to search the hypotheses to find which one matches the current input. We have to do this because the FastPredict shuffles the input batch.
def find_correct_hyp(hyps, input_ids, tokenizer):
    input_tokens = [tokenizer.inv_vocab[id] for id in input_ids]
    dec_input_tokens = input_tokens
    while dec_input_tokens[-1] == '[PAD]':
        dec_input_tokens = dec_input_tokens[:-1]
    if dec_input_tokens[-1] == '[MASK]':
        dec_input_tokens = dec_input_tokens[:-1]
    dec_start_idx = dec_input_tokens.index('[SEP]')+1
    dec_input_tokens = dec_input_tokens[dec_start_idx:]
    for h in hyps:
        if h.tokens == dec_input_tokens:
            return h
    print(h.tokens)
    print(dec_input_tokens)
    print(input_tokens)
    raise Exception('No hypothesis had matching tokens with: [%s]' % ', '.join(dec_input_tokens))

def run_beam_search(decode_one_step_fn, tokenizer, predict_example, beam_size, min_dec_steps, max_dec_steps):
    """Performs beam search decoding on the given example.

    Args:
        sess: a tf.Session
        model: a seq2seq model
        vocab: Vocabulary object
        batch: Batch object that is the same example repeated across the batch

    Returns:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """

    max_dec_steps = max_dec_steps



    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[],
                       log_probs=[],
                       input_example=copy.deepcopy(predict_example),
                       already_added=False,
                       ) for _ in range(beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)


    steps = 0
    while steps < max_dec_steps and results_still_too_small(results, beam_size):

        input_examples = [h.input_example for h in hyps]
        tokens = [h.tokens for h in hyps]
        # Run one step of the decoder to get the new info
        (topk_ids_list, topk_log_probs_list, input_ids_list) = decode_one_step_fn(input_examples, tokens)
        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        for i, h in enumerate(hyps):
            hyp_to_extend = find_correct_hyp(hyps, input_ids_list[i], tokenizer)
            for j in range(beam_size * 2):  # for each of the top 2*beam_size hyps:
                # Extend the ith hypothesis with the jth option
                token_id = topk_ids_list[i][j]
                token = tokenizer.inv_vocab[token_id]
                new_hyp = hyp_to_extend.extend(token=token,
                                   log_prob=topk_log_probs_list[i][j],
                                   )
                all_hyps.append(new_hyp)
            if steps == 0:
                break

        # Filter and collect any hypotheses that have produced the end token.
        hyps = []  # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps):  # in order of most likely h
            if h.latest_token == '[SEP]':  # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= min_dec_steps:
                    results.append(h)
                    h.already_added = True
                    # print 'ADDED THING'
            else:  # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            if len(hyps) == beam_size:
                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop. (Unless it's Logan's better beam search)
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)
    best_hyp = hyps_sorted[0]

    input_tokens = [tokenizer.inv_vocab[id] for id in input_ids_list[0]]
    sep_idx = input_tokens.index('[SEP]')
    input_tokens = input_tokens[:sep_idx]

    # Return the hypothesis with highest average log prob
    return best_hyp, input_tokens


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
