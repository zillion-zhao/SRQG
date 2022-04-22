import requests
import re
import os
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from get_bing_results import parse_result
from utils import read_evaluation_data
from config import *

requests.packages.urllib3.disable_warnings()


# get all HTMLs and extract all lists and texts that related to query or items

class HTMLAnalyzer:
    def __init__(self, url, url_id, path, query, items):
        self.url = url
        self.url_id = url_id
        self.path = path
        self.query = query
        self.items = items
        self.html = self.get_html(url, url_id, path)
        self.text_len_threshold = 60

    def get_html(self, url, url_id, path):
        session = HTMLSession()
        html = session.get(url, timeout=10).text
        with open(path + '/html_' + url_id + '.txt', 'w', encoding='utf-8') as f:
            f.write(html)
        return html

    def html_pre_process(self):
        # remove <script>, <noscript>, <style>, and <!-- --> codes
        re_script = re.compile('<script[\s\S]*?</script>')
        re_noscript = re.compile('<noscript[\s\S]*?</noscript>')
        re_style = re.compile('<style[\s\S]*?</style>')
        re_annotation = re.compile('<!--[\s\S]*?-->')
        self.html = re.sub(re_script, '', self.html)
        self.html = re.sub(re_noscript, '', self.html)
        self.html = re.sub(re_style, '', self.html)
        self.html = re.sub(re_annotation, '', self.html)

        # remove special labels
        self.html = self.html.replace('<b>', '')
        self.html = self.html.replace('</b>', '')
        self.html = self.html.replace('<i>', '')
        self.html = self.html.replace('</i>', '')

        def remove_special(sstr):
            sstr = sstr.group().replace('\n', '').strip()
            results = re.findall('>[\s\S]+?<', sstr)
            for i in range(len(results)):
                results[i] = results[i][1:-1].strip()
            result = ' '.join(results).strip()
            return result

        re_hyper = re.compile('<a [\s\S]*?</a>')
        re_sup = re.compile('<sup [\s\S]*?</sup>')

        self.html = re.sub(re_hyper, remove_special, self.html)  # hyperlink
        self.html = re.sub(re_sup, remove_special, self.html)  # reference

        soup = BeautifulSoup(self.html, "html.parser")
        self.html = soup.prettify()

        with open(self.path + '/html-preprocess_' + self.url_id + '.txt', 'w', encoding='utf-8') as f:
            f.write(self.html)

    def html_get_content(self):
        # remove lines with label
        html_lines = self.html.split('\n')[:-1]
        html_content = []
        for i in range(len(html_lines)):
            # tackle <hn> titles
            if html_lines[i].find('<h1') != -1 or html_lines[i].find('<h2') != -1 or \
                    html_lines[i].find('<h3') != -1 or html_lines[i].find('<h4') != -1 or \
                    html_lines[i].find('<h5') != -1 or html_lines[i].find('<h6') != -1:
                j = i + 1
                while html_lines[j].find('</h') == -1:
                    if html_lines[j].startswith(' '):
                        html_lines[j] = html_lines[j][1:]
                    j += 1

            # remove other all html labels
            if html_lines[i].find('<') != -1 and html_lines[i].find('>') != -1:
                continue
            else:
                html_content.append(html_lines[i])

        with open(self.path + '/html-content_' + self.url_id + '.txt', 'w', encoding='utf-8') as f:
            for i in range(len(html_content)):
                html_content[i] = html_content[i].replace('\t', ' ')
                html_content[i] = html_content[i].replace('* ', ' ')
                html_content[i] = html_content[i].replace('*', ' ')
                html_content[i] = html_content[i].replace('+ ', ' ')
                html_content[i] = html_content[i].replace('+', ' ')
                html_content[i] = html_content[i].replace(' o ', '  ')
                html_content[i] = html_content[i].replace('Â', '')
                html_content[i] = html_content[i].replace('â', '')
                html_content[i] = html_content[i].replace('-', ' ')
                if len(html_content[i].strip()) > 3:  # restrain the length
                    f.write(html_content[i] + '\n')

    def extract_lists_texts_of_url(self):
        # extract lists and texts from HTML content files and save
        lists = []
        texts = []
        lists_index = []  # start position of each list
        texts_index = []  # position of each text
        space_nums = []

        # count space num for each line
        with open(self.path + '/html-content_' + self.url_id + '.txt', 'r', encoding='utf-8') as f:
            contents = f.read().split('\n')[:-1]
        for i in range(len(contents)):
            num = 0
            for j in range(len(contents[i])):
                if contents[i][j] == ' ':
                    num += 1
                else:
                    break
            space_nums.append(num)
            contents[i] = contents[i].strip()

        # extract lists and texts with their indexes
        temp_list = []
        signal = 0
        for i in range(len(contents) - 1):
            if len(contents[i]) >= self.text_len_threshold:  # long enough, as text
                texts.append(contents[i])
                texts_index.append(i)
                continue
            if space_nums[i] == space_nums[i + 1]:
                signal = 1
                temp_list.append(contents[i])
            else:
                if signal == 1:
                    temp_list.append(contents[i])
                    signal = 0
                if len(temp_list) > 1:
                    lists.append(temp_list)
                    lists_index.append(i - len(temp_list) + 1)
                    temp_list = []

        # extract lists titles
        lists_descs = []
        for i in range(len(lists_index)):
            last_1 = contents[lists_index[i] - 1]
            last_2 = contents[lists_index[i] - 2]
            list_space_num = space_nums[lists_index[i]]
            last_1_space_num = space_nums[lists_index[i] - 1]
            last_2_space_num = space_nums[lists_index[i] - 2]
            if last_1_space_num >= list_space_num:
                last_1 = ''
            if last_2_space_num >= list_space_num:
                last_2 = ''

            if last_1 == '' and last_2 == '':
                lists_descs.append('')
            elif last_1 == '' and last_2 != '':
                lists_descs.append(last_2)
            elif last_1 != '' and last_2 == '':
                lists_descs.append(last_1)
            elif len(last_1) <= 60 and len(last_2) <= 60:
                lists_descs.append(', '.join([last_1, last_2]))
            elif len(last_1) <= 60 < len(last_2):
                lists_descs.append(last_1)
            elif len(last_1) > 60 >= len(last_2):
                lists_descs.append(last_2)
            else:
                lists_descs.append('')

        with open(self.path + '/html-lists_texts_' + self.url_id + '.txt', 'w', encoding='utf-8') as f:
            f.write(self.query + '\t' + self.url + '\t' + self.url_id + '\n')
            f.write('-----------------------------------------------------------\n')
            for i in range(len(lists)):
                if lists_descs[i] != '':  # only lists with titles
                    f.write(str(lists_descs[i]) + ' ----- ')
                    for j in range(len(lists[i])):
                        if j != len(lists[i]) - 1:
                            f.write(lists[i][j] + '\t')
                        else:
                            f.write(lists[i][j] + '\n')
            f.write('-----------------------------------------------------------\n')
            for i in range(len(texts)):
                f.write(texts[i] + '\n')
        return lists, lists_descs, texts


class ListExtractor:
    # get lists in html that contains some items
    def __init__(self):
        self.partly_contain_threshold = 0.3

    def partly_contain(self, s, l):
        # s: items set, l: a list
        # if intersection(s, l) >= threshold*len(s) then s(l) partly contains l(s)
        intersection = 0
        for s_i in s:
            for l_i in l:
                s_i_reformat = s_i.replace(' ', '').lower()
                l_i_reformat = l_i.replace(' ', '').lower()
                if l_i_reformat.find(s_i_reformat) != -1:
                    intersection += 1
                    break
        if intersection >= self.partly_contain_threshold * len(s):
            return True
        else:
            return False

    def get_items_candidates_region(self, s, L, L_descs):
        candidate_set = []
        for i in range(len(L)):
            if self.partly_contain(s, L[i]):
                candidate_set.append(L_descs[i])
        return candidate_set


class TextExtractor:
    # get texts in html that contains query or some items
    def __init__(self):
        self.partly_contain_threshold = 0.3

    def partly_contain(self, s, t):
        # s: items set, t: a piece of plain text
        # if threshold of s are contained in t, then return True
        intersection = 0
        t_reformat = t.replace(' ', '').lower()
        for s_i in s:
            s_i_reformat = s_i.replace(' ', '').lower()
            if t_reformat.find(s_i_reformat) != -1:
                intersection += 1
        if intersection >= self.partly_contain_threshold * len(s):
            return True
        else:
            return False

    def get_items_candidates_region(self, s, T):
        candidate_set = []
        for i in range(len(T)):
            if self.partly_contain(s, T[i]):
                candidate_set.append(T[i])
        return candidate_set

    def get_query_candidates_region(self, q, T):
        candidate_set = []
        q_lower = q.lower()
        for i in range(len(T)):
            T_i_lower = T[i].lower()
            if T_i_lower.find(q_lower) != -1:
                candidate_set.append(T[i])
        return candidate_set


def extract_lists_texts_one_url(query, items, url, url_id, top_results_path, query_or_items='query'):
    # extract lists and texts from url for query and items
    query_path = top_results_path + query + '_q/'
    if not os.path.exists(query_path):
        os.mkdir(query_path)
    query_items_path = top_results_path + query + '_qi/'
    if not os.path.exists(query_items_path):
        os.mkdir(query_items_path)
    items_path = top_results_path + query + '_i/'
    if not os.path.exists(items_path):
        os.mkdir(items_path)

    url_id = str(url_id)
    print('url: ' + url + '     ' + 'url_id: ' + url_id, end='')
    if query_or_items == 'query':
        print(' for query')
    elif query_or_items == 'query_items':
        print(' for query and items')
    elif query_or_items == 'items':
        print(' for items')

    query_no_id = query.split('_')[0]

    if query_or_items == 'query':  # only extract texts of query
        # HTML Analyzer
        html_analyzer = HTMLAnalyzer(url, url_id, query_path, query_no_id, items)
        html_analyzer.html_pre_process()
        html_analyzer.html_get_content()
        lists, lists_descs, texts = html_analyzer.extract_lists_texts_of_url()  # get all lists and texts in the url

        # Text Extractor for query
        text_extractor = TextExtractor()
        query_texts_candidates_region = text_extractor.get_query_candidates_region(query_no_id, texts)

        # save text candidates for query
        with open(top_results_path + query + '_q/html-candidates-query_' + url_id + '.txt', 'w', encoding='utf-8') as f:
            for i in range(len(query_texts_candidates_region)):
                if len(query_texts_candidates_region[i]) != 0:
                    f.write(query_texts_candidates_region[i] + '\n')

        return query_texts_candidates_region

    elif query_or_items == 'query_items':  # extract texts of query and text and lists of items
        # HTML Analyzer
        html_analyzer = HTMLAnalyzer(url, url_id, query_items_path, query_no_id, items)
        html_analyzer.html_pre_process()
        html_analyzer.html_get_content()
        lists, lists_descs, texts = html_analyzer.extract_lists_texts_of_url()  # get all lists and texts in the url

        # List Extractor and Text Extractor for items
        list_extractor = ListExtractor()
        text_extractor = TextExtractor()
        query_texts_candidates_region = text_extractor.get_query_candidates_region(query_no_id, texts)
        items_lists_candidates_region = list_extractor.get_items_candidates_region(items, lists, lists_descs)
        items_texts_candidates_region = text_extractor.get_items_candidates_region(items, texts)

        # save text candidates for query
        with open(top_results_path + query + '_qi/html-candidates-query_' + url_id + '.txt', 'w',
                  encoding='utf-8') as f:
            for i in range(len(query_texts_candidates_region)):
                if len(query_texts_candidates_region[i]) != 0:
                    f.write(query_texts_candidates_region[i] + '\n')
        # save list candidates and text candidates for items
        with open(top_results_path + query + '_qi/html-candidates-items_' + url_id + '.txt', 'w',
                  encoding='utf-8') as f:
            for i in range(len(items_lists_candidates_region)):
                if len(items_lists_candidates_region[i]) != 0:
                    f.write(items_lists_candidates_region[i] + '\n')
            f.write('-----------------------------------------------------------\n')
            for i in range(len(items_texts_candidates_region)):
                if len(items_texts_candidates_region[i]) != 0:
                    f.write(items_texts_candidates_region[i] + '\n')

        return query_texts_candidates_region, items_lists_candidates_region, items_texts_candidates_region

    elif query_or_items == 'items':  # extract texts and lists of items
        # HTML Analyzer
        html_analyzer = HTMLAnalyzer(url, url_id, items_path, query_no_id, items)
        html_analyzer.html_pre_process()
        html_analyzer.html_get_content()
        lists, lists_descs, texts = html_analyzer.extract_lists_texts_of_url()  # get all lists and texts in the url

        # List Extractor and Text Extractor for items
        list_extractor = ListExtractor()
        text_extractor = TextExtractor()
        items_lists_candidates_region = list_extractor.get_items_candidates_region(items, lists, lists_descs)
        items_texts_candidates_region = text_extractor.get_items_candidates_region(items, texts)

        # save list candidates and text candidates for items
        with open(top_results_path + query + '_i/html-candidates-items_' + url_id + '.txt', 'w',
                  encoding='utf-8') as f:
            for i in range(len(items_lists_candidates_region)):
                if len(items_lists_candidates_region[i]) != 0:
                    f.write(items_lists_candidates_region[i] + '\n')
            f.write('-----------------------------------------------------------\n')
            for i in range(len(items_texts_candidates_region)):
                if len(items_texts_candidates_region[i]) != 0:
                    f.write(items_texts_candidates_region[i] + '\n')

        return items_lists_candidates_region, items_texts_candidates_region


def extract_lists_texts(query, items, top_results_path):
    # extract lists and texts for query and items in all urls returned by Bing
    print('query: ' + query + '     ' + 'items: ' + str(items))

    if os.path.exists(top_results_path + query + '_qi/candidates-query.txt') and os.path.exists(
            top_results_path + query + '_qi/candidates-items.txt') and os.path.exists(
            top_results_path + query + '_q/candidates-query.txt') and os.path.exists(
            top_results_path + query + '_i/candidates-items.txt'):
        print('lists and texts are already extracted')
        return

    # ------------------------------get all urls for query and its corresponding items------------------------------
    query_urls = []
    query_items_urls = []
    items_urls = []

    # query for search
    file_name = top_results_path + query + '_q_bing_result.json'
    _, _, _, urls = parse_result(file_name)
    query_urls.extend(urls)
    if urls == 'error':
        print('query ', query, ' error')
        return

    # query + all items for search
    file_name = top_results_path + query + '_qi_bing_result.json'
    _, _, _, urls = parse_result(file_name)
    query_items_urls.extend(urls)
    if urls == 'error':
        print('query ', query, ' error')
        return

    # items for search
    file_name = top_results_path + query + '_i_bing_result.json'
    _, _, _, urls = parse_result(file_name)
    items_urls.extend(urls)
    if urls == 'error':
        print('query ', query, ' error')
        return

    # ------------------------------query for search------------------------------
    print('use query for search')
    urls = query_urls
    print(len(urls))
    urls_ids = [str(i) for i in range(len(urls))]

    query_texts_candidates_region_q = []

    for i in range(len(urls)):
        try:
            query_text_region = extract_lists_texts_one_url(query, items, urls[i], urls_ids[i], top_results_path, 'query')
            query_texts_candidates_region_q.extend(query_text_region)
        except Exception:
            continue

    # convert to lower
    for i in range(len(query_texts_candidates_region_q)):
        query_texts_candidates_region_q[i] = query_texts_candidates_region_q[i].lower()

    # ------------------------------query + items for search------------------------------
    print('use query + items for search')
    urls = query_items_urls
    print(len(urls))
    urls_ids = [str(i) for i in range(len(urls))]

    query_texts_candidates_region_qi = []
    items_lists_candidates_region_qi = []
    items_texts_candidates_region_qi = []

    for i in range(len(urls)):
        try:
            query_text_region, items_list_region, items_text_region = extract_lists_texts_one_url(query, items, urls[i], urls_ids[i], top_results_path, 'query_items')
            query_texts_candidates_region_qi.extend(query_text_region)
            items_lists_candidates_region_qi.extend(items_list_region)
            items_texts_candidates_region_qi.extend(items_text_region)
        except Exception:
            continue

    # convert to lower
    for i in range(len(query_texts_candidates_region_qi)):
        query_texts_candidates_region_qi[i] = query_texts_candidates_region_qi[i].lower()
    for i in range(len(items_lists_candidates_region_qi)):
        items_lists_candidates_region_qi[i] = items_lists_candidates_region_qi[i].lower()
    for i in range(len(items_texts_candidates_region_qi)):
        items_texts_candidates_region_qi[i] = items_texts_candidates_region_qi[i].lower()

    # ------------------------------items for search------------------------------
    print('use items for search')
    urls = items_urls
    print(len(urls))
    urls_ids = [str(i) for i in range(len(urls))]

    items_lists_candidates_region_i = []
    items_texts_candidates_region_i = []

    for i in range(len(urls)):
        try:
            items_list_region, items_text_region = extract_lists_texts_one_url(query, items, urls[i], urls_ids[i],
                                                                               top_results_path, 'items')
            items_lists_candidates_region_i.extend(items_list_region)
            items_texts_candidates_region_i.extend(items_text_region)
        except Exception:
            continue

    # convert to lower
    for i in range(len(items_lists_candidates_region_i)):
        items_lists_candidates_region_i[i] = items_lists_candidates_region_i[i].lower()
    for i in range(len(items_texts_candidates_region_i)):
        items_texts_candidates_region_i[i] = items_texts_candidates_region_i[i].lower()

    # save all regions
    with open(top_results_path + query + '_q/candidates-query.txt', 'w', encoding='utf-8') as f:
        f.write(query + '\t' + str(items) + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(query_texts_candidates_region_q)):
            if len(query_texts_candidates_region_q[i]) != 0:
                f.write(query_texts_candidates_region_q[i] + '\n')

    with open(top_results_path + query + '_qi/candidates-query.txt', 'w', encoding='utf-8') as f:
        f.write(query + '\t' + str(items) + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(query_texts_candidates_region_qi)):
            if len(query_texts_candidates_region_qi[i]) != 0:
                f.write(query_texts_candidates_region_qi[i] + '\n')
    with open(top_results_path + query + '_qi/candidates-items.txt', 'w', encoding='utf-8') as f:
        f.write(query + '\t' + str(items) + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(items_lists_candidates_region_qi)):
            if len(items_lists_candidates_region_qi[i]) != 0:
                f.write(items_lists_candidates_region_qi[i] + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(items_texts_candidates_region_qi)):
            if len(items_texts_candidates_region_qi[i]) != 0:
                f.write(items_texts_candidates_region_qi[i] + '\n')

    with open(top_results_path + query + '_i/candidates-items.txt', 'w', encoding='utf-8') as f:
        f.write(query + '\t' + str(items) + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(items_lists_candidates_region_i)):
            if len(items_lists_candidates_region_i[i]) != 0:
                f.write(items_lists_candidates_region_i[i] + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(items_texts_candidates_region_i)):
            if len(items_texts_candidates_region_i[i]) != 0:
                f.write(items_texts_candidates_region_i[i] + '\n')

    # save useful regions to top_results_path (root path)
    with open(top_results_path + query + '_candidates-query.txt', 'w', encoding='utf-8') as f:
        f.write(query + '\t' + str(items) + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(query_texts_candidates_region_q)):
            if len(query_texts_candidates_region_q[i]) != 0:
                f.write(query_texts_candidates_region_q[i] + '\n')
    
    with open(top_results_path + query + '_qi/candidates-items.txt', 'w', encoding='utf-8') as f:
        f.write(query + '\t' + str(items) + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(items_lists_candidates_region_qi)):
            if len(items_lists_candidates_region_qi[i]) != 0:
                f.write(items_lists_candidates_region_qi[i] + '\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(items_texts_candidates_region_qi)):
            if len(items_texts_candidates_region_qi[i]) != 0:
                f.write(items_texts_candidates_region_qi[i] + '\n')

    print('OK')


def extract_eval_lists_texts():
    # read evaluation data
    o_queries, o_items, l_queries, l_items, d_queries, d_items = read_evaluation_data()
    for i in range(100):
        print(i + 1, o_queries[i], o_items[i])
        extract_lists_texts(o_queries[i], o_items[i], top_results_overall_good_path)
    for i in range(100):
        print(i + 1, l_queries[i], l_items[i])
        extract_lists_texts(l_queries[i], l_items[i], top_results_query_log_path)
    for i in range(100):
        print(i + 1, d_queries[i], d_items[i])
        extract_lists_texts(d_queries[i], d_items[i], top_results_query_dimension_path)
