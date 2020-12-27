import copy
import os
import util
import numpy as np


# @profile
def write_highlighted_html(html, out_dir, example_idx):
    html = '''

<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d_highlighted.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d_highlighted.html";
    };

    document.addEventListener("keyup",function(e){
   var key = e.which||e.keyCode;
   switch(key){
      //left arrow
      case 37:
         document.getElementById("btnPrev").click();
      break;
      //right arrow
      case 39:
         document.getElementById("btnNext").click();
      break;
   }
});
</script>

''' % (example_idx - 1, example_idx + 1) + html
    util.create_dirs(out_dir)
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    with open(path, 'w') as f:
        f.write(html)

highlight_colors = ['aqua', 'lime', 'yellow', '#FF7676', '#B9968D', '#D7BDE2', '#8C8DFF', '#D6DBDF', '#F852AF', '#00FF8B', '#FD933A', '#965DFF']
hard_highlight_colors = ['#00BBFF', '#00BB00', '#F4D03F', '#BB5454', '#A16252', '#AF7AC5', '#6668FF', '#AEB6BF', '#FF008F', '#0ECA74', '#FF7400', '#7931FF']
underline_colors = hard_highlight_colors[::-1]
# hard_highlight_colors = ['blue', 'green', 'orange', 'red']

def start_tag_highlight(color, bottom_color=None):
    if bottom_color is not None:
        return "<span style='background: linear-gradient(to top, " + bottom_color + " 50%, " + color + " 50%);'>"
    else:
        return "<span style='background-color: " + color + ";'>"

def start_tag_underline(color):
    return "<span style='border-bottom: 5px solid " + color + ";'>"

def get_idx_for_source_idx(similar_source_indices, source_idx):
    summ_sent_indices = []
    priorities = []
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        for idx_idx, idx in enumerate(source_indices):
            if source_idx == idx:
                summ_sent_indices.append(source_indices_idx)
                priorities.append(idx_idx)
    if len(summ_sent_indices) == 0:
        return None, None
    else:
        return summ_sent_indices, priorities

def determine_start_tag_underline(fusion_locations, sent_idx, token_idx):
    my_chain_idx = -1
    is_done = False
    for chain_idx, chain in enumerate(fusion_locations):
        for loc in chain:
            if sent_idx == loc[0] and token_idx >= loc[1] and token_idx < loc[2]:
                my_chain_idx = chain_idx
                is_done = True
                break
        if is_done:
            break
    if my_chain_idx == -1:
        color = 'white'
    else:
        color = underline_colors[my_chain_idx]
    return start_tag_underline(color)


def html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list,
                                    article_sent_tokens, doc_indices=None, lcs_paths_list=None, article_lcs_paths_list=None, gt_similar_source_indices_list=None,
                                    gt_article_lcs_paths_list=None, fusion_locations=None, summ_fusion_locations=None):
    end_tag = "</span>"
    out_str = ''

    for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
        try:
            similar_source_indices = similar_source_indices_list[summ_sent_idx]
        except:
            similar_source_indices = []
            a=0

        if lcs_paths_list is not None:
            lcs_paths_list = copy.deepcopy(lcs_paths_list)
            lcs_paths = lcs_paths_list[summ_sent_idx]
            if '<br>' in summ_sent:     # this is a special case where we are straight-up copying a pair of sentences as one instance. If this is the case we need to adjust lcs_paths.
                br_idx = summ_sent.index('<br>')
                sent1_len = len(summ_sent[:br_idx])
                lcs_paths[1] = [token_idx+1+sent1_len for token_idx in lcs_paths[1]]

        for token_idx, token in enumerate(summ_sent):
            insert_string = token + ' '
            for source_indices_idx, source_indices in enumerate(similar_source_indices):
                if source_indices_idx == 0:
                    # print summ_sent_idx
                    try:
                        color = hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                    except:
                        print(summ_sent_idx)
                        print(summary_sent_tokens)
                        print('\n')
                else:
                    color = highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                # if token_idx in lcs_paths[source_indices_idx]:
                # if lcs_paths_list is not None:
                #     lcs_paths_list[summ_sent_idx][source_indices_idx]
                if lcs_paths_list is None or token_idx in lcs_paths_list[summ_sent_idx][source_indices_idx]:
                    insert_string = start_tag_highlight(color) + token + ' ' + end_tag
                    break
                # else:
                #     insert_string = start_tag_highlight(highlight_colors[source_indices_idx]) + token + end_tag
                #     break
            out_str += insert_string
        out_str += '<br><br>'

    cur_token_idx = 0
    cur_doc_idx = 0
    for sent_idx, sent in enumerate(article_sent_tokens):
        if doc_indices is not None:
            if cur_token_idx >= len(doc_indices):
                print("Warning: cur_token_idx is greater than len of doc_indices")
            elif doc_indices[cur_token_idx] != cur_doc_idx:
                cur_doc_idx = doc_indices[cur_token_idx]
                out_str += '<br>'
        summ_sent_indices, priorities = get_idx_for_source_idx(similar_source_indices_list, sent_idx)   # summ_sent_indices is what summ_sents are represented in this source sent (e.g. [0, 1, 4]). It's usually 1 summ sent.
                                                                                                        # priorities is 0 if it is the primary article sent. 1 if it is the secondary article sent.
        if priorities is not None:
            colors = [highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
            hard_colors = [hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
        if gt_similar_source_indices_list is not None:
            gt_summ_sent_indices, gt_priorities = get_idx_for_source_idx(gt_similar_source_indices_list, sent_idx)   # summ_sent_indices is what summ_sents are represented in this source sent (e.g. [0, 1, 4]). It's usually 1 summ sent.
                                                                                                        # priorities is 0 if it is the primary article sent. 1 if it is the secondary article sent.
            if gt_priorities is not None:
                gt_colors = [highlight_colors[min(summ_sent_idx, len(highlight_colors) - 1)] for summ_sent_idx in
                             gt_summ_sent_indices]
                gt_hard_colors = [hard_highlight_colors[min(summ_sent_idx, len(highlight_colors) - 1)] for summ_sent_idx in
                                  gt_summ_sent_indices]
        source_sentence = article_sent_tokens[sent_idx]
        out_str += "<p style='margin:5px'>"
        for token_idx, token in enumerate(source_sentence):
            if priorities is None:
                top_color = '#FFFFFF'
            else:
                top_color = '#FFFFFF'
                for priority_idx in reversed(list(range(len(priorities)))):     # priority_idx is the summ_sent_idx
                    summ_sent_idx = summ_sent_indices[priority_idx]
                    priority = priorities[priority_idx]     # priority is 0 if it is the primary article sent. 1 if it is the secondary article sent
                    if article_lcs_paths_list is None or token_idx in article_lcs_paths_list[summ_sent_idx][priority]:
                        top_color = hard_colors[priority_idx] if priority == 0 else colors[priority_idx]
            if gt_article_lcs_paths_list is not None:
                if gt_priorities is None:
                    bottom_color = '#FFFFFF'
                else:
                    bottom_color = '#FFFFFF'
                    for priority_idx in reversed(list(range(len(gt_priorities)))):     # priority_idx is the summ_sent_idx
                        summ_sent_idx = gt_summ_sent_indices[priority_idx]
                        priority = gt_priorities[priority_idx]     # priority is 0 if it is the primary article sent. 1 if it is the secondary article sent
                        if gt_article_lcs_paths_list is not None and token_idx in gt_article_lcs_paths_list[summ_sent_idx][priority]:
                            bottom_color = gt_hard_colors[priority_idx] if priority == 0 else gt_colors[priority_idx]
                start_tag = start_tag_highlight(top_color, bottom_color)
            else:
                start_tag = start_tag_highlight(top_color)
            if fusion_locations is not None:
                start_underline = determine_start_tag_underline(fusion_locations, sent_idx, token_idx)
            else:
                start_underline = '<span>'
            insert_string = start_tag + start_underline + token + ' ' + end_tag + end_tag
            # else:
                # insert_string = start_tag_highlight(highlight_colors[priority]) + token + end_tag
            cur_token_idx += 1
            out_str += insert_string
        # out_str += '<br style="line-height: 400%;">'
        out_str += "</p>"
    out_str += '<br>------------------------------------------------------<br><br>'
    return out_str


def put_html_in_two_columns(html1, html2):
    html = '''
    
    <style>
.row {{
  display: flex;
}}

.column {{
  flex: 50%;
}}
    </style>
    
    <div class="row">
      <div class="column">{}</div>
      <div class="column">{}</div>
    </div>
    '''.format(html1, html2)
    return html


def get_sent_similarities(summ_sent, article_sent_tokens, only_rouge_l=False, remove_stop_words=True):
    similarity_matrix = util.rouge_l_similarity_matrix(article_sent_tokens, [summ_sent], 'recall')
    similarities = np.squeeze(similarity_matrix, 1)

    if not only_rouge_l:
        rouge_l = similarities
        rouge_1 = np.squeeze(util.rouge_1_similarity_matrix(article_sent_tokens, [summ_sent], 'recall', remove_stop_words), 1)
        rouge_2 = np.squeeze(util.rouge_2_similarity_matrix(article_sent_tokens, [summ_sent], 'recall', False), 1)
        similarities = (rouge_l + rouge_1 + rouge_2) / 3.0

    return similarities

def get_simple_source_indices_list(summary_sent_tokens, article_sent_tokens, sentence_limit, min_matched_tokens=2, remove_stop_words=True, lemmatize=True, multiple_ssi=False):
    if lemmatize:
        article_sent_tokens_lemma = util.lemmatize_sent_tokens(article_sent_tokens)
        summary_sent_tokens_lemma = util.lemmatize_sent_tokens(summary_sent_tokens)
    else:
        article_sent_tokens_lemma = article_sent_tokens
        summary_sent_tokens_lemma = summary_sent_tokens

    similar_source_indices_list = []
    lcs_paths_list = []
    article_lcs_paths_list = []
    smooth_article_paths_list = []
    for summ_sent in summary_sent_tokens_lemma:
        remove_lcs = True
        similarities = get_sent_similarities(summ_sent, article_sent_tokens_lemma)
        if remove_lcs:
            similar_source_indices, lcs_paths, article_lcs_paths, smooth_article_paths = get_similar_source_sents_by_lcs(
                summ_sent, summ_sent, list(range(len(summ_sent))), article_sent_tokens_lemma, similarities, 0,
                sentence_limit, min_matched_tokens, remove_stop_words=remove_stop_words, multiple_ssi=multiple_ssi)
            similar_source_indices_list.append(similar_source_indices)
            lcs_paths_list.append(lcs_paths)
            article_lcs_paths_list.append(article_lcs_paths)
            smooth_article_paths_list.append(smooth_article_paths)
    deduplicated_similar_source_indices_list = []
    for sim_source_ind in similar_source_indices_list:
        dedup_sim_source_ind = []
        for ssi in sim_source_ind:
            if not (ssi in dedup_sim_source_ind or ssi[::-1] in dedup_sim_source_ind):
                dedup_sim_source_ind.append(ssi)
        deduplicated_similar_source_indices_list.append(dedup_sim_source_ind)
    # for sim_source_ind_idx, sim_source_ind in enumerate(deduplicated_similar_source_indices_list):
    #     if len(sim_source_ind) > 1:
    #         print ' '.join(summary_sent_tokens[sim_source_ind_idx])
    #         print '-----------'
    #         for ssi in sim_source_ind:
    #             for idx in ssi:
    #                 print ' '.join(article_sent_tokens[idx])
    #             print '-------------'
    #         print '\n\n'
    #         a=0
    simple_similar_source_indices = [tuple(sim_source_ind[0]) for sim_source_ind in deduplicated_similar_source_indices_list]
    lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in lcs_paths_list]
    article_lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in article_lcs_paths_list]
    smooth_article_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in smooth_article_paths_list]
    return simple_similar_source_indices, lcs_paths_list, article_lcs_paths_list, smooth_article_paths_list


def get_similar_source_sents_by_lcs(summ_sent, partial_summ_sent, selection, article_sent_tokens, similarities, depth, sentence_limit, min_matched_tokens, remove_stop_words=True, multiple_ssi=False):
    remove_unigrams = True
    if sentence_limit == 1:
        if depth > 2:
            return [[]], [[]], [[]], [[]]
    elif len(selection) < 3 or depth >= sentence_limit:      # base case: when summary sentence is too short
        return [[]], [[]], [[]], [[]]

    all_sent_indices = []
    all_lcs_paths = []
    all_article_lcs_paths = []
    all_smooth_article_paths = []

    # partial_summ_sent = util.reorder(summ_sent, selection)
    top_sent_indices, top_similarity = get_top_similar_sent(partial_summ_sent, article_sent_tokens, remove_stop_words, multiple_ssi=multiple_ssi)
    top_similarities = util.reorder(similarities, top_sent_indices)
    top_sent_indices = [x for _, x in sorted(zip(top_similarities, top_sent_indices), key=lambda pair: pair[0])][::-1]
    for top_sent_idx in top_sent_indices:
        # top_sent_idx = top_sent_indices[0]
        if remove_unigrams:
            nonstopword_matches, _ = util.matching_unigrams(partial_summ_sent, article_sent_tokens[top_sent_idx], should_remove_stop_words=remove_stop_words)
            lcs_len, (summ_lcs_path, article_lcs_path) = util.matching_unigrams(partial_summ_sent, article_sent_tokens[top_sent_idx])
            smooth_article_path = get_smooth_path(summ_sent, article_sent_tokens[top_sent_idx])
        if len(nonstopword_matches) < min_matched_tokens:
            continue
        # new_selection = [selection[idx] for idx in summ_lcs_path]
        # leftover_selection = [val for idx, val in enumerate(selection) if idx not in summ_lcs_path]
        # partial_summ_sent = replace_with_blanks(summ_sent, leftover_selection)
        leftover_selection = [idx for idx in range(len(partial_summ_sent)) if idx not in summ_lcs_path]
        partial_summ_sent = replace_with_blanks(partial_summ_sent, leftover_selection)

        sent_indices, lcs_paths, article_lcs_paths, smooth_article_paths = get_similar_source_sents_by_lcs(
            summ_sent, partial_summ_sent, leftover_selection, article_sent_tokens, similarities, depth+1,
            sentence_limit, min_matched_tokens, remove_stop_words, multiple_ssi)   # recursive call

        combined_sent_indices = [[top_sent_idx] + indices for indices in sent_indices]      # append my result to the recursive collection
        combined_lcs_paths = [[summ_lcs_path] + paths for paths in lcs_paths]
        combined_article_lcs_paths = [[article_lcs_path] + paths for paths in article_lcs_paths]
        combined_smooth_article_paths = [[smooth_article_path] + paths for paths in smooth_article_paths]

        all_sent_indices.extend(combined_sent_indices)
        all_lcs_paths.extend(combined_lcs_paths)
        all_article_lcs_paths.extend(combined_article_lcs_paths)
        all_smooth_article_paths.extend(combined_smooth_article_paths)
    if len(all_sent_indices) == 0:
        return [[]], [[]], [[]], [[]]
    return all_sent_indices, all_lcs_paths, all_article_lcs_paths, all_smooth_article_paths

def get_smooth_path(summ_sent, article_sent):
    summ_sent = ['<s>'] + summ_sent + ['</s>']
    article_sent = ['<s>'] + article_sent + ['</s>']

    matches = []
    article_indices = []
    summ_token_to_indices = util.create_token_to_indices(summ_sent)
    article_token_to_indices = util.create_token_to_indices(article_sent)
    for key in list(article_token_to_indices.keys()):
        if (util.is_punctuation(key) and not util.is_quotation_mark(key)):
            del article_token_to_indices[key]
    for token in list(summ_token_to_indices.keys()):
        if token in article_token_to_indices:
            article_indices.extend(article_token_to_indices[token])
            matches.extend([token] * len(summ_token_to_indices[token]))
    article_indices = sorted(article_indices)

    # Add a single word or a pair of words if they are in between two hightlighted content words
    new_article_indices = []
    new_article_indices.append(0)
    for article_idx in article_indices[1:]:
        word = article_sent[article_idx]
        prev_highlighted_word = article_sent[new_article_indices[-1]]
        if article_idx - new_article_indices[-1] <= 3 \
                and ((util.is_content_word(word) and util.is_content_word(prev_highlighted_word)) \
                or (len(new_article_indices) >= 2 and util.is_content_word(word) \
                and util.is_content_word(article_sent[new_article_indices[-2]]))):
            in_between_indices = list(range(new_article_indices[-1] + 1, article_idx))
            are_not_punctuation = [not util.is_punctuation(article_sent[in_between_idx]) for in_between_idx in in_between_indices]
            if all(are_not_punctuation):
                new_article_indices.extend(in_between_indices)
        new_article_indices.append(article_idx)
    new_article_indices = new_article_indices[1:-1] # remove <s> and </s> from list

    # Remove isolated stopwords
    new_new_article_indices = []
    for idx, article_idx in enumerate(new_article_indices):
        if (not util.is_stopword_punctuation(article_sent[article_idx])) or (idx > 0 and new_article_indices[idx-1] == article_idx-1) or (idx < len(new_article_indices)-1 and new_article_indices[idx+1] == article_idx+1):
            new_new_article_indices.append(article_idx)
    new_new_article_indices = [idx-1 for idx in new_new_article_indices]    # fix indexing since we don't count <s> and </s>
    return new_new_article_indices

def get_top_similar_sent(summ_sent, article_sent_tokens, remove_stop_words=True, multiple_ssi=False):
    try:
        similarities = get_sent_similarities(summ_sent, article_sent_tokens, remove_stop_words=remove_stop_words)
        top_similarity = np.max(similarities)
    except:
        print(summ_sent)
        print(article_sent_tokens)
        raise
    # sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim == top_similarity]
    if multiple_ssi:
        sent_indices = [sent_idx for sent_idx, sent_sim in enumerate(similarities) if sent_sim > top_similarity * 0.75]
    else:
        sent_indices = [np.argmax(similarities)]
    return sent_indices, top_similarity

def replace_with_blanks(summ_sent, selection):
    replaced_summ_sent = [summ_sent[token_idx] if token_idx in selection else '' for token_idx, token in enumerate(summ_sent)]
    return  replaced_summ_sent

def list_labels_from_probs(sys_token_probs_list, binarize_method, binarize_paramater):
    token_tags_list = token_probs_to_binary_tags(sys_token_probs_list, binarize_method, binarize_paramater)
    article_lcs_paths_list = [binary_tags_to_list(token_tags) for token_tags in token_tags_list]
    return article_lcs_paths_list

def token_probs_to_binary_tags(sys_token_probs_list, binarize_method, binarize_paramater):
    if binarize_method == 'threshold':
        threshold = binarize_paramater
        token_tags_list = [[[1 if score >= threshold else 0 for score in sent] for sent in inst] for inst in sys_token_probs_list]
    elif binarize_method == 'summ_limit':
        limit = binarize_paramater
        sys_token_probs_list_flat = util.flatten_list_of_lists_3d(sys_token_probs_list)
        if binarize_paramater <= 1:
            limit = int(len(sys_token_probs_list_flat) * binarize_paramater)
        else:
            limit = int(binarize_paramater)
        token_tags = [0] * len(sys_token_probs_list_flat)
        for _ in range(limit):
            my_max = -1
            my_argmax = -1
            for score_idx, score in enumerate(sys_token_probs_list_flat):
                is_already_tagged = token_tags[score_idx]
                if not is_already_tagged and score > my_max:
                    my_max = score
                    my_argmax = score_idx
            if my_argmax == -1:     # Stop if a max is not found. This is when we've added all the tokens already.
                break
            else:
                token_tags[my_argmax] = 1
        token_tags_list = util.reshape_like_3d(token_tags, sys_token_probs_list)
    elif binarize_method == 'inst_limit':
        token_tags_list = []
        for token_probs in sys_token_probs_list:
            token_probs_flat = util.flatten_list_of_lists(token_probs)
            if binarize_paramater <= 1:
                limit = int(len(token_probs_flat) * binarize_paramater)
            else:
                limit = int(binarize_paramater)
            token_tags = [0] * len(token_probs_flat)
            indices = np.argsort(token_probs_flat)[-limit:]
            for idx in indices:
                token_tags[idx] = 1
            token_tags = util.reshape_like(token_tags, token_probs)
            token_tags_list.append(token_tags)
    return token_tags_list


def binary_tags_to_list(token_tags):
    article_lcs_paths = []
    for sent in token_tags:
        path = [token_idx for token_idx, tag in enumerate(sent) if tag == 1]
        article_lcs_paths.append(path)
    return article_lcs_paths



















