import os
import json
import tkinter as tk  # GUI
from tkinter import messagebox
from utils import read_evaluation_data
from config import *

# this .py file must be run on a computer that supports a graphical interface


def load_labeling():
    o_queries, o_items, l_queries, l_items, d_queries, d_items = read_evaluation_data()  # load evaluation data

    # tackle the situations that the json files do not exist
    if 'labeling_overall_good.json' not in os.listdir(EVAL_DATA_PATH):
        print('overall good no json file, building dictionaries')
        overall_good_dict = {}  # {query: [items, {query_candidates: scores}, {items_candidates: scores}]}
        for i in range(100):
            query = o_queries[i]
            items = o_items[i]
            if o_queries[i] not in overall_good_dict.keys():
                overall_good_dict[query] = []
                overall_good_dict[query].append(items)
                overall_good_dict[query].append({})
                overall_good_dict[query].append({})
        json_str = json.dumps(overall_good_dict)
        with open(EVAL_DATA_PATH + 'labeling_overall_good.json', 'w', encoding='utf-8') as f:
            f.write(json_str)

    if 'labeling_query_log.json' not in os.listdir(EVAL_DATA_PATH):
        print('query log no json file, building dictionaries')
        query_log_dict = {}  # {query: [items, {query_candidates: scores}, {items_candidates: scores}]}
        for i in range(100):
            query = l_queries[i]
            items = l_items[i]
            if l_queries[i] not in query_log_dict.keys():
                query_log_dict[query] = []
                query_log_dict[query].append(items)
                query_log_dict[query].append({})
                query_log_dict[query].append({})
        json_str = json.dumps(query_log_dict)
        with open(EVAL_DATA_PATH + 'labeling_query_log.json', 'w', encoding='utf-8') as f:
            f.write(json_str)

    if 'labeling_query_dimension.json' not in os.listdir(EVAL_DATA_PATH):
        print('query_dimension no json file, building dictionaries')
        query_dimension_dict = {}  # {query: [items, {query_candidates: scores}, {items_candidates: scores}]}
        for i in range(100):
            query = d_queries[i]
            items = d_items[i]
            if d_queries[i] not in query_dimension_dict.keys():
                query_dimension_dict[query] = []
                query_dimension_dict[query].append(items)
                query_dimension_dict[query].append({})
                query_dimension_dict[query].append({})
        json_str = json.dumps(query_dimension_dict)
        with open(EVAL_DATA_PATH + 'labeling_query_dimension.json', 'w', encoding='utf-8') as f:
            f.write(json_str)

    if 'labeling_overall_good.json' in os.listdir(EVAL_DATA_PATH) and 'labeling_query_log.json' in os.listdir(
            EVAL_DATA_PATH) and 'labeling_query_dimension.json' in os.listdir(EVAL_DATA_PATH):
        print('json files already exist, loading')
        with open(EVAL_DATA_PATH + 'labeling_overall_good.json', 'r', encoding='utf-8') as f:
            overall_good_dict = json.load(f)
        with open(EVAL_DATA_PATH + 'labeling_query_log.json', 'r', encoding='utf-8') as f:
            query_log_dict = json.load(f)
        with open(EVAL_DATA_PATH + 'labeling_query_dimension.json', 'r', encoding='utf-8') as f:
            query_dimension_dict = json.load(f)

    return overall_good_dict, query_log_dict, query_dimension_dict


def show_window():
    # load query, items and labeling data from files
    overall_good_dict, log_dict, dimension_dict = load_labeling()
    flag = -1  # 0 for query dimension, 1 for query log, 2 for overall good, -1 for none

    # create a window object
    window = tk.Tk()
    window.title('Clarifying Question Data Labeling Tool')
    window.geometry('800x325')  # size of window

    label_query = tk.Label(window, width=7, text='query:')
    label_query.place(x=10, y=18)

    input_query = tk.Entry(window, show=None, width=50)
    input_query.place(x=70, y=20)


    def hit_button_query():
        global flag
        query = input_query.get()
        if query == '':
            messagebox.showwarning('empty query', 'please input a query')
        else:
            # clear all entries and lists
            input_items.delete(0, 'end')
            list_query.delete(0, 'end')
            list_items.delete(0, 'end')

            if query in overall_good_dict.keys():
                flag = 2
                items = overall_good_dict[query][0]
                input_items.insert('0', ', '.join(items))
                query_candidates_dict = overall_good_dict[query][1]
                items_candidates_dict = overall_good_dict[query][2]
                for qc in query_candidates_dict.keys():
                    list_query.insert('end', qc + ' --- ' + str(query_candidates_dict[qc]))
                for ic in items_candidates_dict.keys():
                    list_items.insert('end', ic + ' --- ' + str(items_candidates_dict[ic]))
            elif query in log_dict.keys():
                flag = 1
                items = log_dict[query][0]
                input_items.insert('0', ', '.join(items))
                query_candidates_dict = log_dict[query][1]
                items_candidates_dict = log_dict[query][2]
                for qc in query_candidates_dict.keys():
                    list_query.insert('end', qc + ' --- ' + str(query_candidates_dict[qc]))
                for ic in items_candidates_dict.keys():
                    list_items.insert('end', ic + ' --- ' + str(items_candidates_dict[ic]))
            elif query in dimension_dict.keys():
                flag = 0
                items = dimension_dict[query][0]
                input_items.insert('0', ', '.join(items))
                query_candidates_dict = dimension_dict[query][1]
                items_candidates_dict = dimension_dict[query][2]
                for qc in query_candidates_dict.keys():
                    list_query.insert('end', qc + ' --- ' + str(query_candidates_dict[qc]))
                for ic in items_candidates_dict.keys():
                    list_items.insert('end', ic + ' --- ' + str(items_candidates_dict[ic]))
            else:
                messagebox.showwarning('wrong query', 'query not in evaluation data')


    button_query = tk.Button(window, text='submit', width=10, command=hit_button_query)
    button_query.place(x=440, y=15)

    label_items = tk.Label(window, width=7, text='items:')
    label_items.place(x=10, y=58)

    input_items = tk.Entry(window, show=None, width=100)
    input_items.place(x=70, y=60)

    # ----------------------------------------candidates of query description----------------------------------------
    label_query_candidates = tk.Label(window, width=14, text='query candidates:')
    label_query_candidates.place(x=18, y=100)

    list_query = tk.Listbox(window, selectmode=tk.SINGLE)
    list_query.place(x=18, y=120)


    def read_qc():
        try:
            selected = str(list_query.get(list_query.curselection())).split(' --- ')
            input_query_desc.delete(0, 'end')
            input_query_freq.delete(0, 'end')
            input_query_desc.insert('0', selected[0])
            input_query_freq.insert('0', selected[1])
        except Exception:
            messagebox.showwarning('not chosen', 'please choose a query candidate')


    def update_qc():
        query = input_query.get()
        desc = input_query_desc.get()
        freq = input_query_freq.get()

        if flag == 2:
            if desc not in overall_good_dict[query][1].keys():
                messagebox.showwarning('new candidate', 'new candidate, please click add button')
            else:
                overall_good_dict[query][1][desc] = freq  # update dictionary
                json_str = json.dumps(overall_good_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_overall_good.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_query.delete(0, 'end')
                for qc in overall_good_dict[query][1].keys():
                    list_query.insert('end', qc + ' --- ' + str(overall_good_dict[query][1][qc]))

        elif flag == 1:
            if desc not in log_dict[query][1].keys():
                messagebox.showwarning('new candidate', 'new candidate, please click add button')
            else:
                log_dict[query][1][desc] = freq  # update dictionary
                json_str = json.dumps(log_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_log.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_query.delete(0, 'end')
                for qc in log_dict[query][1].keys():
                    list_query.insert('end', qc + ' --- ' + str(log_dict[query][1][qc]))

        elif flag == 0:
            if desc not in dimension_dict[query][1].keys():
                messagebox.showwarning('new candidate', 'new candidate, please click add button')
            else:
                dimension_dict[query][1][desc] = freq  # update dictionary
                json_str = json.dumps(dimension_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_dimension.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_query.delete(0, 'end')
                for qc in dimension_dict[query][1].keys():
                    list_query.insert('end', qc + ' --- ' + str(dimension_dict[query][1][qc]))


    def add_qc():
        query = input_query.get()
        desc = input_query_desc.get()
        freq = input_query_freq.get()

        if flag == 2:
            if desc in overall_good_dict[query][1].keys():
                messagebox.showwarning('candidate exists', 'candidate exists, please click update button')
            else:
                overall_good_dict[query][1][desc] = freq  # add it into dictionary
                json_str = json.dumps(overall_good_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_overall_good.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_query.delete(0, 'end')
                for qc in overall_good_dict[query][1].keys():
                    list_query.insert('end', qc + ' --- ' + str(overall_good_dict[query][1][qc]))

        elif flag == 1:
            if desc in log_dict[query][1].keys():
                messagebox.showwarning('candidate exists', 'candidate exists, please click update button')
            else:
                log_dict[query][1][desc] = freq  # add it into dictionary
                json_str = json.dumps(log_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_log.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_query.delete(0, 'end')
                for qc in log_dict[query][1].keys():
                    list_query.insert('end', qc + ' --- ' + str(log_dict[query][1][qc]))

        elif flag == 0:
            if desc in dimension_dict[query][1].keys():
                messagebox.showwarning('candidate exists', 'candidate exists, please click update button')
            else:
                dimension_dict[query][1][desc] = freq  # add it into dictionary
                json_str = json.dumps(dimension_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_dimension.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_query.delete(0, 'end')
                for qc in dimension_dict[query][1].keys():
                    list_query.insert('end', qc + ' --- ' + str(dimension_dict[query][1][qc]))


    button_read_qc = tk.Button(window, text='read qc', width=9, command=read_qc)
    button_read_qc.place(x=400, y=100)
    button_update_qc = tk.Button(window, text='update qc', width=11, command=update_qc)
    button_update_qc.place(x=520, y=100)
    button_add_qc = tk.Button(window, text='add qc', width=8, command=add_qc)
    button_add_qc.place(x=660, y=100)

    input_query_desc = tk.Entry(window, show=None, width=30)
    input_query_desc.place(x=400, y=155)
    input_query_freq = tk.Entry(window, show=None, width=10)
    input_query_freq.place(x=650, y=155)

    # ----------------------------------------candidates of items description----------------------------------------
    label_items_candidates = tk.Label(window, width=14, text='items candidates:')
    label_items_candidates.place(x=220, y=100)

    list_items = tk.Listbox(window, selectmode=tk.SINGLE)
    list_items.place(x=220, y=120)


    def read_ic():
        try:
            selected = str(list_items.get(list_items.curselection())).split(' --- ')
            input_items_desc.delete(0, 'end')
            input_items_freq.delete(0, 'end')
            input_items_desc.insert('0', selected[0])
            input_items_freq.insert('0', selected[1])
        except Exception:
            messagebox.showwarning('not chosen', 'please choose an items candidate')


    def update_ic():
        query = input_query.get()
        desc = input_items_desc.get()
        freq = input_items_freq.get()

        if flag == 2:
            if desc not in overall_good_dict[query][2].keys():
                messagebox.showwarning('new candidate', 'new candidate, please click add button')
            else:
                overall_good_dict[query][2][desc] = freq  # update dictionary
                json_str = json.dumps(overall_good_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_overall_good.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_items.delete(0, 'end')
                for qc in overall_good_dict[query][2].keys():
                    list_items.insert('end', qc + ' --- ' + str(overall_good_dict[query][2][qc]))

        elif flag == 1:
            if desc not in log_dict[query][2].keys():
                messagebox.showwarning('new candidate', 'new candidate, please click add button')
            else:
                log_dict[query][2][desc] = freq  # update dictionary
                json_str = json.dumps(log_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_log.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_items.delete(0, 'end')
                for qc in log_dict[query][2].keys():
                    list_items.insert('end', qc + ' --- ' + str(log_dict[query][2][qc]))

        elif flag == 0:
            if desc not in dimension_dict[query][2].keys():
                messagebox.showwarning('new candidate', 'new candidate, please click add button')
            else:
                dimension_dict[query][2][desc] = freq  # update dictionary
                json_str = json.dumps(dimension_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_dimension.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_items.delete(0, 'end')
                for qc in dimension_dict[query][2].keys():
                    list_items.insert('end', qc + ' --- ' + str(dimension_dict[query][2][qc]))


    def add_ic():
        query = input_query.get()
        desc = input_items_desc.get()
        freq = input_items_freq.get()

        if flag == 2:
            if desc in overall_good_dict[query][2].keys():
                messagebox.showwarning('candidate exists', 'candidate exists, please click update button')
            else:
                overall_good_dict[query][2][desc] = freq  # update dictionary
                json_str = json.dumps(overall_good_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_overall_good.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_items.delete(0, 'end')
                for qc in overall_good_dict[query][2].keys():
                    list_items.insert('end', qc + ' --- ' + str(overall_good_dict[query][2][qc]))

        elif flag == 1:
            if desc in log_dict[query][2].keys():
                messagebox.showwarning('candidate exists', 'candidate exists, please click update button')
            else:
                log_dict[query][2][desc] = freq  # update dictionary
                json_str = json.dumps(log_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_log.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_items.delete(0, 'end')
                for qc in log_dict[query][2].keys():
                    list_items.insert('end', qc + ' --- ' + str(log_dict[query][2][qc]))

        elif flag == 0:
            if desc in dimension_dict[query][2].keys():
                messagebox.showwarning('candidate exists', 'candidate exists, please click update button')
            else:
                dimension_dict[query][2][desc] = freq  # update dictionary
                json_str = json.dumps(dimension_dict)  # write dictionary into json file
                with open(EVAL_DATA_PATH + 'labeling_query_dimension.json', 'w', encoding='utf-8') as f:
                    f.write(json_str)

                list_items.delete(0, 'end')
                for qc in dimension_dict[query][2].keys():
                    list_items.insert('end', qc + ' --- ' + str(dimension_dict[query][2][qc]))


    button_read_ic = tk.Button(window, text='read ic', width=9, command=read_ic)
    button_read_ic.place(x=400, y=200)
    button_update_ic = tk.Button(window, text='update ic', width=11, command=update_ic)
    button_update_ic.place(x=520, y=200)
    button_add_ic = tk.Button(window, text='add ic', width=8, command=add_ic)
    button_add_ic.place(x=660, y=200)

    input_items_desc = tk.Entry(window, show=None, width=30)
    input_items_desc.place(x=400, y=255)
    input_items_freq = tk.Entry(window, show=None, width=10)
    input_items_freq.place(x=650, y=255)

    window.mainloop()
