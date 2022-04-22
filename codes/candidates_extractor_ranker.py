import re
import os
import inflect
import stanza
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from utils import tanh, all_in, distance_score, read_evaluation_data, cos_sim
from config import *

# candidates extractor and ranker: extracts description candidates from extracted lists and texts files
# then ranks these description candidates according to several human-designed features

# find top 5 descriptions of a query or an item from WebIsA
class WebisaExtractor:
    def __init__(self):
        self.top_count = 5

    def list_sort(self, descs, freqs):
        # sort descriptions and frequencies by frequencies
        length = len(descs)
        for i in range(length - 1):
            for j in range(length - 1 - i):
                if freqs[j] < freqs[j + 1]:
                    t = freqs[j]
                    freqs[j] = freqs[j + 1]
                    freqs[j + 1] = t
                    t = descs[j]
                    descs[j] = descs[j + 1]
                    descs[j + 1] = t
        return descs[:self.top_count], freqs[:self.top_count]

    def find_desc(self, term):
        initial_char = term[0]
        descs = []
        freqs = []
        with open(webisa_path + initial_char + '_ten.txt', 'r') as fp:
            content = fp.read().split('\n')[:-1]
            for i in range(len(content)):
                content[i] = content[i].split('\t')
                if content[i][0] == term:
                    descs.append(content[i][1])
                    freqs.append(float(content[i][2]))
        return self.list_sort(descs, freqs)


# find top 5 descriptions of a query or an item from Concept Graph
class ConceptGraphExtractor:
    def __init__(self):
        self.top_count = 5

    def list_sort(self, descs, freqs):
        # sort descriptions and frequencies by frequencies
        length = len(descs)
        for i in range(length - 1):
            for j in range(length - 1 - i):
                if freqs[j] < freqs[j + 1]:
                    t = freqs[j]
                    freqs[j] = freqs[j + 1]
                    freqs[j + 1] = t
                    t = descs[j]
                    descs[j] = descs[j + 1]
                    descs[j + 1] = t
        return descs[:self.top_count], freqs[:self.top_count]

    def find_desc(self, term):
        with open(concept_graph_path, 'r', encoding='utf-8') as f:
            content = f.read().split('\n')[:-1]
        descs = []
        freqs = []
        for i in range(len(content)):
            content[i] = content[i].split('\t')
            if content[i][1] == term:
                descs.append(content[i][0])
                freqs.append(int(content[i][2]))
        return self.list_sort(descs, freqs)


class CandidatesExtractorRanker:
    def __init__(self, query, items, top_results_path, full_feature=False):
        print('build a candidates extractor and ranker')
        self.query = query
        self.query_no_id = query.split('_')[0]
        self.items = items
        self.top_results_path = top_results_path
        self.full_feature = full_feature

        self.items_candidates_texts = []
        self.items_candidates_lists = []
        self.query_candidates_texts = []

        # get lists and texts for items
        with open(top_results_path + query + '_candidates-items.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        candidates = content.split('\n')[:-1]

        flag = 0
        for i in range(len(candidates)):
            # for judging lists lines (1) or text lines (2)
            if candidates[i] == '-----------------------------------------------------------':
                flag += 1
            candidates[i] = self.preprocess(candidates[i])
            if 2 < len(candidates[i]) <= 30 and flag == 1:  # from lists
                self.items_candidates_lists.append(candidates[i])
            elif len(candidates[i]) > 30 and flag == 2:  # from texts
                self.items_candidates_texts.append(candidates[i])

        new_lists = []
        # split by ','
        for i in range(len(self.items_candidates_lists)):
            temp_list = self.items_candidates_lists[i].split(',')
            new_lists.extend(temp_list)
        self.items_candidates_lists = new_lists

        for i in range(len(self.items_candidates_lists)):
            # remove special characters
            self.items_candidates_lists[i] = self.items_candidates_lists[i].replace('.', '')
            self.items_candidates_lists[i] = self.items_candidates_lists[i].replace('!', '')
            self.items_candidates_lists[i] = self.items_candidates_lists[i].replace('?', '')

        # get texts for query
        with open(top_results_path + query + '_candidates-query.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        candidates = content.split('\n')[:-1]

        for i in range(len(candidates)):
            candidates[i] = self.preprocess(candidates[i])
            if len(candidates[i]) > 40:
                self.query_candidates_texts.append(candidates[i])

        self.inflector = inflect.engine()  # convert nouns to singular: self.inflector.singular_noun(word)
        self.parser = stanza.Pipeline('en', stanza_path, use_gpu=True)  # Stanford parser

        # read stopwords
        with open(stopword_path, 'r') as f:
            self.stopwords = f.read().split('\n')[:-1]

        # candidates
        self.query_candidates = []
        self.items_candidates = []

        # temp data
        self.title_tuples = []

        self.webisa_query_descs = []
        self.webisa_query_freqs = []
        self.concept_query_descs = []
        self.concept_query_freqs = []

        self.webisa_items_descs = []
        self.webisa_items_freqs = []
        self.concept_items_descs = []
        self.concept_items_freqs = []

        # scale parameters
        # list statistical feature
        self.p_items_list = 0.3

        # text statistical feature
        self.p_query_pattern = 1.0
        self.p_items_pattern = 1.0
        self.p_query_distance = 2.0
        self.p_items_distance = 2.0
        self.p_query_occur = 1.0
        self.p_items_occur = 1.0
        self.p_query_freq = 0.1
        self.p_items_freq = 0.1
        self.p_query_inclusion = 0.5
        self.p_items_inclusion = 0.5

        # semantic feature
        self.p_query_semantic = 1.0
        self.p_items_semantic = 1.0

        # entity feature
        self.p_query_entity = 0.1
        self.p_items_entity = 0.1

        # inhibition feature
        self.p_items_sim = 10.0
        self.tau_is = 0.95

    def preprocess(self, line):
        # remove special characters, just maintain numbers and alphabets
        line = line.replace('&amp;', 'and')
        line = line.replace(' ', ' ')
        line_process = re.sub(r'[^A-Za-z0-9 .,?!\']+', '', line)

        # convert multiple spaces to just one
        line_process = ' '.join(line_process.split())
        line_process = line_process.strip()
        return line_process

    def get_query_candidates(self):
        print('getting query candidates')
        # candidates of query description are tuples extracted from patterns
        # 1. patterns
        with open(isa_pattern_path, 'r', encoding='utf-8') as f:
            patterns = f.read().split('\n')[:-1]
        all_matched_NPhs = []
        all_matched_NPts = []
        for line in self.query_candidates_texts:
            if line.find('[') != -1:
                continue
            line_singular = convert_singular_line(self.inflector, line)
            for pattern in patterns:
                match_str = pattern.replace('NPt', '').replace('NPh', '')  # is a, such as, et al.
                pos_nph = pattern.find('NPh')
                pos_npt = pattern.find('NPt')
                # match patterns in each text line
                pos_match = [each.start() for each in re.finditer(match_str, line_singular)]
                for pos in pos_match:
                    start_pos = pos - 1
                    start_space_num = 0
                    while start_space_num < 10 and start_pos > 0:
                        start_pos -= 1
                        if line_singular[start_pos] == ' ':
                            start_space_num += 1

                    end_pos = pos + len(match_str) + 1
                    end_space_num = 0
                    while end_space_num < 10 and end_pos < len(line_singular) - 1:
                        end_pos += 1
                        if line_singular[end_pos] == ' ':
                            end_space_num += 1

                    if start_pos < 0:
                        start_pos = 0
                    if end_pos >= len(line_singular):
                        end_pos = len(line_singular)

                    # NPt is a NPh
                    if pos_nph > pos_npt and line_singular[start_pos:pos].find(self.query_no_id) != -1:
                        all_matched_NPhs.append(line_singular[pos + len(match_str):end_pos + 1])
                        all_matched_NPts.append(line_singular[start_pos:pos])
                    # NPh such as NPt
                    elif pos_nph < pos_npt and line_singular[pos + len(match_str):end_pos + 1].find(self.query_no_id) != -1:
                        all_matched_NPhs.append(line_singular[start_pos:pos])
                        all_matched_NPts.append(line_singular[pos + len(match_str):end_pos + 1])

        nph_tuples = self.get_freq_dicts(all_matched_NPhs)
        nph_tuples = combine_freq_dicts(self.query_no_id, self.items, nph_tuples[0], nph_tuples[1], nph_tuples[2],
                                        nph_tuples[3], nph_tuples[4], self.inflector)
        nph_tuples = process_freq_dict(nph_tuples, self.inflector, self.query_no_id, self.items, self.parser)

        for i in range(len(nph_tuples)):
            # if a candidate of query contains any item or the query itself, it is deemed not a good candidate
            good = 1
            for item in self.items:
                if nph_tuples[i][0].find(item) != -1:
                    good = 0
                    break
            if good == 1 and nph_tuples[i][0] != self.query_no_id:
                candidate = re.sub(r'[^A-Za-z0-9 ]+', '', nph_tuples[i][0])
                if candidate.find(self.query_no_id) == -1:
                    self.query_candidates.append(candidate)

        # 2. get query candidates from webisa data and concept graph data
        webisa_extractor = WebisaExtractor()
        webisa_query_descs, webisa_query_freqs = webisa_extractor.find_desc(self.query_no_id)
        concept_graph_extractor = ConceptGraphExtractor()
        concept_query_descs, concept_query_freqs = concept_graph_extractor.find_desc(self.query_no_id)

        self.webisa_query_descs = webisa_query_descs
        self.webisa_query_freqs = webisa_query_freqs
        self.concept_query_descs = concept_query_descs
        self.concept_query_freqs = concept_query_freqs

        for i in range(len(webisa_query_descs)):
            candidate = re.sub(r'[^A-Za-z0-9 ]+', '', webisa_query_descs[i])
            if candidate.find(self.query_no_id) == -1:
                self.query_candidates.append(candidate)
        for i in range(len(concept_query_descs)):
            candidate = re.sub(r'[^A-Za-z0-9 ]+', '', concept_query_descs[i])
            if candidate.find(self.query_no_id) == -1:
                self.query_candidates.append(candidate)

        self.query_candidates = list(set(self.query_candidates))
        print('query candidates:', len(self.query_candidates))
        print(self.query_candidates)

    def get_items_candidates(self):
        print('getting items candidates')
        # candidates of items description are tuples extracted from list titles and items text contexts
        # 1. list titles
        title_tuples = self.get_freq_dicts(self.items_candidates_lists)
        title_tuples = combine_freq_dicts(self.query_no_id, self.items, title_tuples[0], title_tuples[1],
                                          title_tuples[2], title_tuples[3], title_tuples[4], self.inflector)
        title_tuples = process_freq_dict(title_tuples, self.inflector, self.query_no_id, self.items, self.parser)

        self.title_tuples = title_tuples
        for i in range(len(title_tuples)):
            # if a candidate contains any item, it is deemed not a good candidate
            good = 1
            for item in self.items:
                if title_tuples[i][0].find(item) != -1:
                    good = 0
                    break
            if good == 1:
                candidate = re.sub(r'[^A-Za-z0-9 ]+', '', title_tuples[i][0])
                if candidate not in self.items:
                    self.items_candidates.append(candidate)

        # 2. items text contexts
        all_contexts = []
        context_range = 50
        for line in self.items_candidates_texts:
            if line.find('[') != -1:
                continue
            line_singular = convert_singular_line(self.inflector, line)
            for item in self.items:
                pos_s = [each.start() for each in re.finditer(item, line_singular)]
                for pos in pos_s:
                    start_pos = pos - context_range
                    if start_pos < 0:
                        start_pos = 0
                    end_pos = pos + len(item) + context_range
                    if end_pos >= len(line_singular):
                        end_pos = len(line_singular)
                    all_contexts.append(line_singular[start_pos:end_pos])

        context_tuples = self.get_freq_dicts(all_contexts)
        context_tuples = combine_freq_dicts(self.query_no_id, self.items, context_tuples[0], context_tuples[1],
                                            context_tuples[2], context_tuples[3], context_tuples[4], self.inflector)
        context_tuples = process_freq_dict(context_tuples, self.inflector, self.query_no_id, self.items, self.parser)

        for i in range(len(context_tuples)):
            # if a candidate contains any item, it is deemed not a good candidate
            good = 1
            for item in self.items:
                if context_tuples[i][0].find(item) != -1:
                    good = 0
                    break
            if good == 1:
                candidate = re.sub(r'[^A-Za-z0-9 ]+', '', context_tuples[i][0])
                if candidate not in self.items:
                    self.items_candidates.append(candidate)

        # 3. get items candidates from webisa data and concept graph data
        webisa_extractor = WebisaExtractor()
        concept_graph_extractor = ConceptGraphExtractor()

        for item in self.items:
            webisa_item_descs, webisa_item_freqs = webisa_extractor.find_desc(item)
            concept_item_descs, concept_item_freqs = concept_graph_extractor.find_desc(item)

            for i in range(len(webisa_item_descs)):
                candidate = re.sub(r'[^A-Za-z0-9 ]+', '', webisa_item_descs[i])
                if candidate not in self.items:
                    self.items_candidates.append(candidate)
            for i in range(len(concept_item_descs)):
                candidate = re.sub(r'[^A-Za-z0-9 ]+', '', concept_item_descs[i])
                if candidate not in self.items:
                    self.items_candidates.append(candidate)

            self.webisa_items_descs.append(webisa_item_descs)
            self.webisa_items_freqs.append(webisa_item_freqs)
            self.concept_items_descs.append(concept_item_descs)
            self.concept_items_freqs.append(concept_item_freqs)

        self.items_candidates = list(set(self.items_candidates))
        print('items candidates:', len(self.items_candidates))
        print(self.items_candidates)

    def get_freq_dicts(self, candidates_texts):
        # candidates_lines: items_candidates_lines or query_candidates_lines
        all_words = []
        all_tuples = []
        all_triples = []
        all_4ples = []
        all_5ples = []
        for line in candidates_texts:
            if line.find('[') != -1:
                continue
            words = line.split(' ')
            for i in range(len(words)):
                words[i] = re.sub(r'[^A-Za-z0-9 ]+', '', words[i])

            # get words (without stopwords)
            for word in words:
                if word.startswith('-') or word in self.stopwords:
                    continue
                all_words.append(word)
            # get tuples
            for i in range(len(words) - 1):
                all_tuples.append(words[i] + ' ' + words[i + 1])
            # get triples
            for i in range(len(words) - 2):
                all_triples.append(words[i] + ' ' + words[i + 1] + ' ' + words[i + 2])
            # get 4ples
            for i in range(len(words) - 3):
                all_4ples.append(words[i] + ' ' + words[i + 1] + ' ' + words[i + 2] + ' ' + words[i + 3])
            # get 5ples
            for i in range(len(words) - 4):
                all_5ples.append(
                    words[i] + ' ' + words[i + 1] + ' ' + words[i + 2] + ' ' + words[i + 3] + ' ' + words[i + 4])

        dic_words = {}
        for word in all_words:
            if word not in dic_words:
                dic_words[word] = 1
            else:
                dic_words[word] += 1
        sorted_dic_words = sorted(dic_words.items(), key=lambda dic_words: dic_words[1], reverse=True)

        dic_tuples = {}
        for word in all_tuples:
            if word not in dic_tuples:
                dic_tuples[word] = 1
            else:
                dic_tuples[word] += 1
        sorted_dic_tuples = sorted(dic_tuples.items(), key=lambda dic_tuples: dic_tuples[1], reverse=True)

        dic_triples = {}
        for word in all_triples:
            if word not in dic_triples:
                dic_triples[word] = 1
            else:
                dic_triples[word] += 1
        sorted_dic_triples = sorted(dic_triples.items(), key=lambda dic_triples: dic_triples[1], reverse=True)

        dic_4ples = {}
        for word in all_4ples:
            if word not in dic_4ples:
                dic_4ples[word] = 1
            else:
                dic_4ples[word] += 1
        sorted_dic_4ples = sorted(dic_4ples.items(), key=lambda dic_4ples: dic_4ples[1], reverse=True)

        dic_5ples = {}
        for word in all_5ples:
            if word not in dic_5ples:
                dic_5ples[word] = 1
            else:
                dic_5ples[word] += 1
        sorted_dic_5ples = sorted(dic_5ples.items(), key=lambda dic_5ples: dic_5ples[1], reverse=True)

        return sorted_dic_words, sorted_dic_tuples, sorted_dic_triples, sorted_dic_4ples, sorted_dic_5ples

    # --------------------list statistical features--------------------
    def get_list_title_features(self, items_candidates):
        print('getting list title features')
        title_tuples = self.title_tuples
        items_features = []
        for candidate in items_candidates:
            appended = 0
            for i in range(len(title_tuples)):
                if title_tuples[i][0] == candidate:
                    items_features.append(float(title_tuples[i][1]))
                    appended = 1
                    break
            if not appended:
                items_features.append(0.0)

        if not self.full_feature:
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_list * items_features[i])

        return items_features

    # --------------------text statistical features--------------------
    def get_pattern_features(self, query_candidates, items_candidates):
        print('getting pattern features')
        # read patterns
        with open(isa_pattern_path, 'r', encoding='utf-8') as f:
            patterns = f.read().split('\n')[:-1]

        query_matched_NPhs = []
        query_matched_NPts = []
        items_matched_NPhs = []
        items_matched_NPts = []

        query_features = []
        items_features = []

        for line in self.query_candidates_texts:
            if line.find('[') != -1:
                continue
            line_singular = convert_singular_line(self.inflector, line)
            for pattern in patterns:
                match_str = pattern.replace('NPt', '').replace('NPh', '')  # is a, such as, et al.
                pos_nph = pattern.find('NPh')
                pos_npt = pattern.find('NPt')
                # match patterns in each text line
                pos_match = [each.start() for each in re.finditer(match_str, line_singular)]
                for pos in pos_match:
                    start_pos = pos - 1
                    start_space_num = 0
                    while start_space_num < 10 and start_pos > 0:
                        start_pos -= 1
                        if line_singular[start_pos] == ' ':
                            start_space_num += 1

                    end_pos = pos + len(match_str) + 1
                    end_space_num = 0
                    while end_space_num < 10 and end_pos < len(line_singular) - 1:
                        end_pos += 1
                        if line_singular[end_pos] == ' ':
                            end_space_num += 1

                    if start_pos < 0:
                        start_pos = 0
                    if end_pos >= len(line_singular):
                        end_pos = len(line_singular)
                    # for query
                    # NPt is a NPh
                    if pos_nph > pos_npt and line_singular[start_pos:pos].find(self.query_no_id) != -1:
                        query_matched_NPhs.append(line_singular[pos + len(match_str):end_pos + 1])
                        query_matched_NPts.append(line_singular[start_pos:pos])
                    # NPh such as NPt
                    elif pos_nph < pos_npt and line_singular[pos + len(match_str):end_pos + 1].find(self.query_no_id) != -1:
                        query_matched_NPhs.append(line_singular[start_pos:pos])
                        query_matched_NPts.append(line_singular[pos + len(match_str):end_pos + 1])

        for line in self.items_candidates_texts:
            if line.find('[') != -1:
                continue
            line_singular = convert_singular_line(self.inflector, line)
            for pattern in patterns:
                match_str = pattern.replace('NPt', '').replace('NPh', '')  # is a, such as, et al.
                pos_nph = pattern.find('NPh')
                pos_npt = pattern.find('NPt')
                # match patterns in each text line
                pos_match = [each.start() for each in re.finditer(match_str, line_singular)]
                for pos in pos_match:
                    start_pos = pos - 40
                    end_pos = pos + len(match_str) + 40
                    if start_pos < 0:
                        start_pos = 0
                    if end_pos >= len(line_singular):
                        end_pos = len(line_singular)

                    # for items
                    for item in self.items:
                        # NPt is a NPh
                        if pos_nph > pos_npt and line_singular[start_pos:pos].find(item) != -1:
                            items_matched_NPhs.append(line_singular[pos + len(match_str):end_pos + 1])
                            items_matched_NPts.append(line_singular[start_pos:pos])
                        # NPh such as NPt
                        elif pos_nph < pos_npt and line_singular[pos + len(match_str):end_pos + 1].find(item) != -1:
                            items_matched_NPhs.append(line_singular[start_pos:pos])
                            items_matched_NPts.append(line_singular[pos + len(match_str):end_pos + 1])

        for q_candidate in query_candidates:
            q_feature = 0.0
            for i in range(len(query_matched_NPhs)):
                if query_matched_NPhs[i].find(q_candidate) != -1 and query_matched_NPts[i].find(self.query_no_id) != -1:
                    q_feature += 1.0
            query_features.append(q_feature)

        for i_candidate in items_candidates:
            i_feature = 0.0
            for i in range(len(items_matched_NPhs)):
                if items_matched_NPhs[i].find(i_candidate) != -1:
                    items_num = 0
                    for item in self.items:
                        if items_matched_NPts[i].find(item) != -1:
                            items_num += 1
                    items_rate = items_num / len(self.items)
                    i_feature += (1.0 * items_rate)
            items_features.append(i_feature)

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_pattern * query_features[i])
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_pattern * items_features[i])

        return query_features, items_features

    def get_distance_features(self, query_candidates, items_candidates):
        print('getting distance features')
        query_all_contexts = []
        items_all_contexts = []
        query_pos_s = []  # start pos of each query in query_all_contexts
        items_pos_s = []  # start pos of each item in items_all_contexts
        context_range = 50

        query_features = []
        items_features = []

        for line in self.query_candidates_texts:
            if line.find('[') != -1:
                continue
            line_singular = convert_singular_line(self.inflector, line)
            pos_query_line = [each.start() for each in re.finditer(self.query_no_id, line_singular)]
            for pos in pos_query_line:
                start_pos = pos - context_range
                if start_pos < 0:
                    start_pos = 0
                end_pos = pos + len(self.query_no_id) + context_range
                if end_pos >= len(line_singular):
                    end_pos = len(line_singular)
                query_pos_s.append(pos - start_pos)
                query_all_contexts.append(line_singular[start_pos:end_pos])

        for line in self.items_candidates_texts:
            if line.find('[') != -1:
                continue
            line_singular = convert_singular_line(self.inflector, line)

            for item in self.items:
                pos_item_line = [each.start() for each in re.finditer(item, line_singular)]
                for pos in pos_item_line:
                    start_pos = pos - context_range
                    if start_pos < 0:
                        start_pos = 0
                    end_pos = pos + len(item) + context_range
                    if end_pos >= len(line_singular):
                        end_pos = len(line_singular)
                    items_pos_s.append(pos - start_pos)
                    items_all_contexts.append(line_singular[start_pos:end_pos])

        for q_candidate in query_candidates:
            q_feature = 0.0
            for i in range(len(query_all_contexts)):
                # find q_candidate from query_all_contexts[i]
                pos_s = [each.start() for each in re.finditer(q_candidate, query_all_contexts[i])]
                for pos in pos_s:
                    dist = distance_score(pos, query_pos_s[i])
                    q_feature += dist
            query_features.append(q_feature)

        for i_candidate in items_candidates:
            i_feature = 0.0
            for i in range(len(items_all_contexts)):
                # find i_candidate from items_all_contexts[i]
                pos_s = [each.start() for each in re.finditer(i_candidate, items_all_contexts[i])]
                for pos in pos_s:
                    dist = distance_score(pos, items_pos_s[i])
                    i_feature += dist
            items_features.append(i_feature)

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_distance * query_features[i])
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_distance * items_features[i])

        return query_features, items_features

    def get_co_occur_features(self, query_candidates, items_candidates):
        print('getting co-occurrence features')
        query_features = [0.0] * len(query_candidates)
        items_features = [0.0] * len(items_candidates)

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_occur * query_features[i])
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_occur * items_features[i])

        return query_features, items_features

    def get_freq_features(self, query_candidates, items_candidates):
        print('getting frequency features')
        query_features = []
        items_features = []
        for q_candidate in query_candidates:
            q_feature = 0.0
            for i in range(len(self.query_candidates_texts)):
                q_feature += self.query_candidates_texts[i].count(q_candidate)
            query_features.append(q_feature)

        for i_candidate in items_candidates:
            i_feature = 0.0
            for i in range(len(self.items_candidates_texts)):
                i_feature += self.items_candidates_texts[i].count(i_candidate)
            for i in range(len(self.items_candidates_lists)):
                i_feature += self.items_candidates_lists[i].count(i_candidate)
            items_features.append(i_feature)

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_freq * query_features[i])
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_freq * items_features[i])

        return query_features, items_features

    def get_inclusion_features(self, query_candidates, items_candidates):
        print('getting inclusion features')
        query_features = []
        items_features = []

        for i in range(len(query_candidates)):
            q_feature = 0.0
            for j in range(len(query_candidates)):
                if i != j:
                    if query_candidates[i].find(query_candidates[j]) != -1:
                        q_feature += 1.0
            query_features.append(q_feature)

        for i in range(len(items_candidates)):
            i_feature = 0.0
            for j in range(len(items_candidates)):
                if i != j:
                    if items_candidates[i].find(items_candidates[j]) != -1:
                        i_feature += 1.0
            items_features.append(i_feature)

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_inclusion * query_features[i])
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_inclusion * items_features[i])

        return query_features, items_features

    # --------------------semantic features--------------------
    def get_bert_features(self, query_candidates, items_candidates):
        print('getting contextual semantic features')

        return [0.0] * len(query_candidates), [0.0] * len(items_candidates)

        tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        model = TFAutoModel.from_pretrained(bert_model_path)

        # get encoding of query
        query_input = tokenizer(self.query_no_id, return_tensors='tf', padding=True)
        query_encoding = model(query_input)

        # get encoding of items
        items_input = tokenizer(self.items, return_tensors='tf', padding=True)
        items_encoding = model(items_input)

        # get encoding of query candidates
        query_candidates_input = tokenizer(query_candidates, return_tensors='tf', padding=True)
        query_candidates_encoding = model(query_candidates_input)

        # get encoding of items candidates
        items_candidates_input = tokenizer(items_candidates, return_tensors='tf', padding=True)
        items_candidates_encoding = model(items_candidates_input)
        # print('items candidates encoding of each token: ', items_candidates_encoding[0].numpy().shape)
        # print('items candidates encoding', items_candidates_encoding[1].numpy().shape)

        query_features = []
        items_features = []

        # add semantic features
        for i in range(len(query_candidates)):
            q_feature = cos_sim(query_encoding[1][0], query_candidates_encoding[1][i])
            query_features.append(q_feature)

        for i in range(len(items_candidates)):
            i_feature = 0.0
            for j in range(len(self.items)):
                i_feature += cos_sim(items_encoding[1][j], items_candidates_encoding[1][i])
            items_features.append(i_feature / len(self.items))

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_semantic * query_features[i])
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_semantic * items_features[i])

        return query_features, items_features

    # --------------------entity features--------------------
    def get_entity_features(self, query_candidates, items_candidates):
        print('getting entity features')
        webisa_extractor = WebisaExtractor()
        concept_graph_extractor = ConceptGraphExtractor()

        webisa_query_descs, webisa_query_freqs = webisa_extractor.find_desc(self.query_no_id)
        webisa_items_descs = []
        webisa_items_freqs = []
        for item in self.items:
            webisa_item_descs, webisa_item_freqs = webisa_extractor.find_desc(item)
            webisa_items_descs.append(webisa_item_descs)
            webisa_items_freqs.append(webisa_item_freqs)

        concept_query_descs, concept_query_freqs = concept_graph_extractor.find_desc(self.query_no_id)
        concept_items_descs = []
        concept_items_freqs = []
        for item in self.items:
            concept_item_descs, concept_item_freqs = concept_graph_extractor.find_desc(item)
            concept_items_descs.extend(concept_item_descs)
            concept_items_freqs.extend(concept_item_freqs)

        query_features = []
        items_features = []

        for i in range(len(query_candidates)):
            q_feature = 0.0
            for j in range(len(webisa_query_descs)):
                if query_candidates[i] == webisa_query_descs[j]:
                    q_feature += webisa_query_freqs[j]
            for j in range(len(concept_query_descs)):
                if query_candidates[i] == concept_query_descs[j]:
                    q_feature += concept_query_freqs[j]
            query_features.append(q_feature)

        for i in range(len(items_candidates)):
            i_feature = 0.0
            for j in range(len(webisa_items_descs)):
                if items_candidates[i] == webisa_items_descs[j]:
                    i_feature += webisa_items_freqs[j]
            for j in range(len(concept_items_descs)):
                if items_candidates[i] == concept_items_descs[j]:
                    i_feature += concept_items_freqs[j]
            items_features.append(i_feature)

        if not self.full_feature:
            for i in range(len(query_features)):
                query_features[i] = tanh(self.p_query_entity * query_features[i])
                if np.isnan(query_features[i]):
                    query_features[i] = 1.0
            for i in range(len(items_features)):
                items_features[i] = tanh(self.p_items_entity * items_features[i])
                if np.isnan(items_features[i]):
                    items_features[i] = 1.0

        return query_features, items_features

    # --------------------inhibition features--------------------
    def get_items_sim_features(self, items_candidates):

        return [0.0] * len(items_candidates)

        tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        model = TFAutoModel.from_pretrained(bert_model_path)

        items_input = tokenizer(self.items, return_tensors='tf', padding=True)
        items_encoding = model(items_input)

        count = 0
        sum_cos = 0.0
        for i in range(len(items_encoding[1].numpy())):
            for j in range(len(items_encoding[1].numpy())):
                if i < j:
                    count += 1
                    sum_cos += cos_sim(items_encoding[1].numpy()[i], items_encoding[1].numpy()[j])
        sum_cos /= count
        print(sum_cos)
        if sum_cos >= self.tau_is:
            return [0.0] * len(items_candidates)
        else:
            return [self.p_items_sim * (self.tau_is - sum_cos)] * len(items_candidates)


def convert_singular_line(inflector, line):
    words_with_s = ['this', 'as', 'is', 'news', 'windows', 'virus', 'supernoobs', 'does', 'os', 'ios', 'macos', 'pus',
                    'bus', 'vs', 'ps', 'js', 'ls', 'us', 'cs', 'kiss', 'miss', 'ms', 'nds', 'nes', 'class', 'mass',
                    'his', 'its', 'guess', 'success', 'business', 'happiness', 'abscess', 'across', 'has', 'diagnosis',
                    'dress']
    words = line.split(' ')
    singular = []
    for word in words:
        if inflector.singular_noun(word) and word not in words_with_s:
            singular.append(inflector.singular_noun(word))
        else:
            singular.append(word)
    return ' '.join(singular)


def combine_freq_dicts(query, items, words, tuples, triples, fourples, fiveples, inflector):
    query_split = query.split(' ')
    items_split = []
    for item in items:
        items_split.extend(item.split(' '))

    words_processed = []
    tuples_processed = []
    triples_processed = []
    fourples_processed = []
    fiveples_processed = []

    for i in range(len(words)):
        if not (words[i][0] == '' or words[i][0].isdigit() or query.find(words[i][0]) != -1 or ' '.join(items).find(
                words[i][0]) != -1):
            words_processed.append(words[i])

    for i in range(len(tuples)):
        if not (all_in(tuples[i][0].split(' '), query_split + items_split)):
            tuples_processed.append(tuples[i])
    for i in range(len(triples)):
        if not (all_in(triples[i][0].split(' '), query_split + items_split)):
            triples_processed.append(triples[i])
    for i in range(len(fourples)):
        if not (all_in(fourples[i][0].split(' '), query_split + items_split)):
            fourples_processed.append(fourples[i])
    for i in range(len(fiveples)):
        if not (all_in(fiveples[i][0].split(' '), query_split + items_split)):
            fiveples_processed.append(fiveples[i])

    tuples = words_processed[:30] + tuples_processed[:30] + triples_processed[:30] + \
             fourples_processed[:30] + fiveples_processed[:30]

    # convert to singular
    processed_tuples = []
    for i in range(len(tuples)):
        line = tuples[i][0]
        line_singular = convert_singular_line(inflector, line)
        processed_tuples.append((line_singular, tuples[i][1]))

    # re-ranking
    combined_tuples = {}
    for i in range(len(processed_tuples)):
        tup, freq = processed_tuples[i]
        if tup not in combined_tuples.keys():
            combined_tuples[tup] = freq
        else:
            combined_tuples[tup] += freq
    combined_tuples = sorted(combined_tuples.items(), key=lambda combined_tuples: combined_tuples[1], reverse=True)

    return combined_tuples


def process_freq_dict(tuples, inflector, query, items, parser):
    # convert to singular
    singular_tuples = []
    for i in range(len(tuples)):
        line = tuples[i][0]
        line_singular = convert_singular_line(inflector, line).strip()
        singular_tuples.append((line_singular, tuples[i][1]))

    # remove useless tuples
    removed_tuples = []
    for i in range(len(singular_tuples)):
        if not (singular_tuples[i][0] == '' or singular_tuples[i][0].isdigit() or query.find(
                singular_tuples[i][0]) != -1 or ' '.join(items).find(singular_tuples[i][0]) != -1):
            removed_tuples.append(singular_tuples[i])

    # re-ranking
    combined_tuples = {}
    for i in range(len(removed_tuples)):
        tup, freq = removed_tuples[i]
        if tup not in combined_tuples.keys():
            combined_tuples[tup] = freq
        else:
            combined_tuples[tup] += freq
    combined_tuples = sorted(combined_tuples.items(), key=lambda combined_tuples: combined_tuples[1], reverse=True)

    # remove tuples without nouns
    removed_nouns_tuples = []
    for tuple in combined_tuples:
        doc = parser(tuple[0])
        tuple_words = []
        tuple_pos = []
        for word in doc.sentences[0].words:
            text, pos = word.text, word.pos  # parts-of-speech
            tuple_words.append(text)
            tuple_pos.append(pos)
        if 'NOUN' in tuple_pos and (tuple_pos[-1] == 'NOUN' or tuple_pos[-1] == 'PROPN') and \
                (tuple_pos[0] == 'NOUN' or tuple_pos[0] == 'PROPN' or tuple_pos[0] == 'ADJ'):
            removed_nouns_tuples.append(tuple)

    return removed_nouns_tuples


class QueryCandidateFeatures:
    def __init__(self, query_candidate, pattern_f, distance_f, occur_f, freq_f, inc_f, semantic_f, entity_f):
        self.query_candidate = query_candidate

        self.pattern_f = pattern_f
        self.distance_f = distance_f
        self.occur_f = occur_f
        self.freq_f = freq_f
        self.inc_f = inc_f

        self.semantic_f = semantic_f

        self.entity_f = entity_f

        self.score = pattern_f + distance_f + occur_f + freq_f + inc_f + semantic_f + entity_f


class ItemsCandidateFeatures:
    def __init__(self, items_candidate, list_f, pattern_f, distance_f, occur_f, freq_f, inc_f, semantic_f, entity_f, items_sim_f):
        self.items_candidate = items_candidate

        self.list_f = list_f

        self.pattern_f = pattern_f
        self.distance_f = distance_f
        self.occur_f = occur_f
        self.freq_f = freq_f
        self.inc_f = inc_f

        self.semantic_f = semantic_f

        self.entity_f = entity_f

        self.items_sim_f = items_sim_f

        self.score = list_f + pattern_f + distance_f + occur_f + freq_f + inc_f + semantic_f + entity_f - items_sim_f


def extract_rank_candidates(query, items, top_results_path, result_save_path, full_feature):
    # get different types of features and combine them together to score each candidates
    if os.path.exists(result_save_path + query + '_query.txt') or os.path.exists(result_save_path + query + '_items.txt'):
        print('result already exists')
        return

    candidates_ranker = CandidatesExtractorRanker(query, items, top_results_path, full_feature)
    # candidates for query and items, save as candidates_ranker.query_candidates and candidates_ranker.items_candidates
    candidates_ranker.get_query_candidates()
    candidates_ranker.get_items_candidates()

    # get feature scores
    # --------------------list statistical features--------------------
    items_list_title_features = candidates_ranker.get_list_title_features(candidates_ranker.items_candidates)
    print('items list title features:', len(items_list_title_features))

    # --------------------text statistical features--------------------
    # pattern feature
    query_pattern_features, items_pattern_features = candidates_ranker.get_pattern_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items pattern features:', len(query_pattern_features), len(items_pattern_features))
    # distance feature
    query_distance_features, items_distance_features = candidates_ranker.get_distance_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items distance features:', len(query_distance_features), len(items_distance_features))
    # co-occurrence feature
    query_occur_features, items_occur_features = candidates_ranker.get_co_occur_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items occurrence features:', len(query_occur_features), len(items_occur_features))
    # frequency feature
    query_freq_features, items_freq_features = candidates_ranker.get_freq_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items frequency features:', len(query_freq_features), len(items_freq_features))
    # inclusion feature
    query_inc_features, items_inc_features = candidates_ranker.get_inclusion_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items inclusion features:', len(query_inc_features), len(items_inc_features))

    # --------------------semantic features--------------------
    query_bert_features, items_bert_features = candidates_ranker.get_bert_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items semantics features:', len(query_bert_features), len(items_bert_features))

    # --------------------entity features--------------------
    query_entity_features, items_entity_features = candidates_ranker.get_entity_features(candidates_ranker.query_candidates, candidates_ranker.items_candidates)
    print('query /items entity features:', len(query_entity_features), len(items_entity_features))

    # --------------------inhibition features--------------------
    # items sim feature
    items_sim_features = candidates_ranker.get_items_sim_features(candidates_ranker.items_candidates)
    print('items sim inhibition features:', len(items_sim_features))

    # combine all features and calculate score of each candidate
    print('--------------------query candidates--------------------')
    query_candidate_features = []
    for i in range(len(candidates_ranker.query_candidates)):
        query_candidate_features.append(QueryCandidateFeatures(candidates_ranker.query_candidates[i], query_pattern_features[i], query_distance_features[i],
                                                               query_occur_features[i], query_freq_features[i], query_inc_features[i], query_bert_features[i],
                                                               query_entity_features[i]))

    # sort query_candidate_features by score from big to small
    query_candidate_features.sort(key=lambda x: x.score, reverse=True)

    with open(result_save_path + query + '_query.txt', 'a', encoding='utf-8') as f:
        for i in range(len(query_candidate_features)):
            print(query_candidate_features[i].query_candidate.ljust(50, ' ') + '\t' +
                  str(format(query_candidate_features[i].pattern_f, '.4f')) + '\t' +
                  str(format(query_candidate_features[i].distance_f, '.4f')) + '\t' +
                  str(format(query_candidate_features[i].occur_f, '.4f')) + '\t' +
                  str(format(query_candidate_features[i].freq_f, '.4f')) + '\t' +
                  str(format(query_candidate_features[i].inc_f, '.4f')) + '\t' + '|' + '\t' +
                  str(format(query_candidate_features[i].semantic_f, '.4f')) + '\t' + '|' + '\t' +
                  str(format(query_candidate_features[i].entity_f, '.4f')))

            f.write(query_candidate_features[i].query_candidate.ljust(50, ' ') + '\t' +
                    str(format(query_candidate_features[i].pattern_f, '.4f')) + '\t' +
                    str(format(query_candidate_features[i].distance_f, '.4f')) + '\t' +
                    str(format(query_candidate_features[i].occur_f, '.4f')) + '\t' +
                    str(format(query_candidate_features[i].freq_f, '.4f')) + '\t' +
                    str(format(query_candidate_features[i].inc_f, '.4f')) + '\t' + '|' + '\t' +
                    str(format(query_candidate_features[i].semantic_f, '.4f')) + '\t' + '|' + '\t' +
                    str(format(query_candidate_features[i].entity_f, '.4f')) + '\t\n')

    print('--------------------items candidates--------------------')
    items_candidate_features = []
    for i in range(len(candidates_ranker.items_candidates)):
        items_candidate_features.append(ItemsCandidateFeatures(candidates_ranker.items_candidates[i], items_list_title_features[i], items_pattern_features[i],
                                                               items_distance_features[i], items_occur_features[i], items_freq_features[i], items_inc_features[i],
                                                               items_bert_features[i], items_entity_features[i], items_sim_features[i]))

    # sort items_candidate_features by score from big to small
    items_candidate_features.sort(key=lambda x: x.score, reverse=True)

    with open(result_save_path + query + '_items.txt', 'a', encoding='utf-8') as f:
        for i in range(len(items_candidate_features)):
            print(items_candidate_features[i].items_candidate.ljust(50, ' ') + '\t' +
                  str(format(items_candidate_features[i].list_f, '.4f')) + '\t' + '|' + '\t' +
                  str(format(items_candidate_features[i].pattern_f, '.4f')) + '\t' +
                  str(format(items_candidate_features[i].distance_f, '.4f')) + '\t' +
                  str(format(items_candidate_features[i].occur_f, '.4f')) + '\t' +
                  str(format(items_candidate_features[i].freq_f, '.4f')) + '\t' +
                  str(format(items_candidate_features[i].inc_f, '.4f')) + '\t' + '|' + '\t' +
                  str(format(items_candidate_features[i].semantic_f, '.4f')) + '\t' + '|' + '\t' +
                  str(format(items_candidate_features[i].entity_f, '.4f')) + '\t' + '|' + '\t' + 
                  str(format(-items_candidate_features[i].items_sim_f, '.4f')))

            f.write(items_candidate_features[i].items_candidate.ljust(50, ' ') + '\t' +
                    str(format(items_candidate_features[i].list_f, '.4f')) + '\t' + '|' + '\t' +
                    str(format(items_candidate_features[i].pattern_f, '.4f')) + '\t' +
                    str(format(items_candidate_features[i].distance_f, '.4f')) + '\t' +
                    str(format(items_candidate_features[i].occur_f, '.4f')) + '\t' +
                    str(format(items_candidate_features[i].freq_f, '.4f')) + '\t' +
                    str(format(items_candidate_features[i].inc_f, '.4f')) + '\t' + '|' + '\t' +
                    str(format(items_candidate_features[i].semantic_f, '.4f')) + '\t' + '|' + '\t' +
                    str(format(items_candidate_features[i].entity_f, '.4f')) + '\t' + '|' + '\t' + 
                    str(format(-items_candidate_features[i].items_sim_f, '.4f')) + '\t\n')


def extract_rank_evaluation_candidates(full_feature=False):
    # read evaluation data
    o_queries, o_items, l_queries, l_items, d_queries, d_items = read_evaluation_data()
    for i in range(100):
        print(i + 1, o_queries[i], o_items[i])
        try:
            extract_rank_candidates(o_queries[i], o_items[i], top_results_overall_good_path, overall_good_candidates_path, full_feature)
        except Exception:
            print('Error')
            continue
    for i in range(100):
        print(i + 1, l_queries[i], l_items[i])
        try:
            extract_rank_candidates(l_queries[i], l_items[i], top_results_query_log_path, query_log_candidates_path, full_feature)
        except Exception:
            print('Error')
            continue
    for i in range(100):
        print(i + 1, d_queries[i], d_items[i])
        try:
            extract_rank_candidates(d_queries[i], d_items[i], top_results_query_dimension_path, query_dimension_candidates_path, full_feature)
        except Exception:
            print('Error')
            continue


def extract_rank_srqg_ltr_candidates(full_feature=False):
    # read training data
    query_set = []
    items_set = []
    with open(srqg_ltr_training_data_path, 'r', encoding='utf-8') as f:
        contents = f.read().split('\n')[:-1]
        for i, line in enumerate(contents):
            line_split = line.split('\t')
            query = line_split[0]
            items = eval(line_split[1])
            query_set.append(query)
            items_set.append(items)

    for i in range(0, 400):
        print(i + 1, query_set[i], items_set[i])
        try:
            extract_rank_candidates(query_set[i], items_set[i], top_results_srqg_ltr_path, srqg_ltr_candidates_path, full_feature)
        except Exception:
            print('Error')
            continue


def extract_rank_srqg_gen_candidates(full_feature=False):
    # read training data
    query_set = []
    items_set = []
    inflector = inflect.engine()
    with open(qlm_data_5_build_path, 'r', encoding='utf-8') as f:
        contents = f.read().split('\n')[:-1]
        for i, line in enumerate(contents):
            line_split = line.split('\t')
            query = line_split[0].split(' <sep> ')[0]
            items = []
            items_strs = line_split[1].split(' <itemssep> ')
            for sstr in items_strs:
                item = sstr.split(' <sep> ')[0]
                if item != '<unknown>':
                    items.append(convert_singular_line(inflector, item))
            query_set.append(query)
            items_set.append(items)
    print(len(query_set))
    print(len(items_set))

    for i in range(len(query_set)):  # len(query_set)
        print(i + 1, query_set[i], items_set[i])
        try:
            extract_rank_candidates(query_set[i], items_set[i], top_results_srqg_gen_path, srqg_gen_candidates_path, full_feature)
        except Exception:
            print('Error')
            continue
