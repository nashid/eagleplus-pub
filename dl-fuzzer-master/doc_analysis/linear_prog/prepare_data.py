from typing import Pattern
from numpy.core.numeric import True_
from numpy.lib.financial import irr
from util import *
from nltk.parse.corenlp import CoreNLPDependencyParser
from mining_util import *
import pandas as pd
import random
import argparse
import collections

# TODO: merge `structure` and `tensor_t` col
constr_cols = ['dtype', 'structure', 'shape', 'ndim', 'enum', 'range']

def get_sent_idx(df, candidate_idx, sample_ratio, remove_noconstr_sent=True):
    assert len(list(set(candidate_idx))) == len(candidate_idx)
    keep_rows = []
    if not remove_noconstr_sent:
        keep_rows =  candidate_idx
    else:
        for index in candidate_idx:
            nullness = df.iloc[index].isnull()
            include = False
            for constr_col in constr_cols:
                if not nullness[constr_col]:    # nulless=False, constr_col isn't null
                    include=True            # include this line
                    break

            if include: 
                keep_rows.append(index)
    if sample_ratio<1 and sample_ratio>0:
        sample_size = int(len(keep_rows) * sample_ratio)
        keep_rows = random.choices(keep_rows, k=sample_size)
    return keep_rows

def prepare_sent(df, send_idx, col='Normalized_descp', lower=False):
    if not isinstance(df.iloc[send_idx][col], str):
        print(df.iloc[send_idx][col])
        sent = 'one_word none'
    else:
        sent = df.iloc[send_idx][col]


    if lower:
        sent = sent.lower()
    return sent

# class IR():
#     def __init__(self, category, value):
#         self.category = category
#         self.value = value
#         assert self.category in constr_cols

# def init_ir_record():
#     ret = {}
#     for cat in constr_cols:
#         ret[cat] = set()
#     return ret


# def parse_ir(row):
#     def _constr_empty(constr_row):
#         for cat in constr_row:
#             if constr_row[cat]:
#                 return False
#         return True
    
#     constr_row = init_ir_record()
#     nulless = row.isnull()
#     for cat in constr_cols:
#         if not nulless[cat]: 
#             for c in row[cat].split(';'):
#                 constr_row[cat].add(c)

#     assert not _constr_empty(constr_row)
#     # assert (filter_by_anno or not _constr_empty(constr_row))    # if filter_by_anno=False, the constr_row shoundn't be empty
#     return constr_row

def check_key_cont(map):
    # check the key of the `map` is continuous
    for i in range(len(map)):
        assert i in map

def check_app_non_empty(app_map):
    for idx, app in app_map.items():
        assert app

def count_overlap(l1, l2):
    return len([x for x in l1 if x in l2])


def run(sample_path, save_path, max_iter, sample_ratio=1):
    # variables required:
    # - subtree_map: a dict maps <subtree idx>(i) to <subtree>
    # - reverse_subtree_map: a dict maps <subtree> to <subtree idx>(i)
    # - ir_map: a dict maps <IR idx>(j) to tuple (IR category, IR val)
    # - reverse_ir_map: a nested dict {`dtype`: {<IR val>: <IR idx>}, `structure`: {...}}
    #
    # - subtree_app: appearence record, a dict maps <subtree idx> to <a list of sentence idx> it appears in
    # - ir_app: appearence record, a dict maps <IR idx> to <a list of sentence idx> it appears in
    # 
    # - m: number of IR, len(ir_map)
    # - n: number of subtree pattern, len(subtree_app)
    # - cooccur_cnt: 2d list, n*m, the count of co-occurence
    # - prob: 2d list, n*m, can be calculated with cooccur_cnt and subtree_app
    # - subtree_size: list of length n, size of each subtree pattern
    
    # init 
    subtree_map = {}
    reverse_subtree_map = {}
    ir_map = {}
    reverse_ir_map = {}
    for cat in constr_cols:
        reverse_ir_map[cat] = {}
    subtree_app = collections.defaultdict(list)
    ir_app = collections.defaultdict(list)
    subtree_size = {}


    df = pd.read_csv(sample_path)       # get csv into df
    candidate_idx = list(range(len(df)))    # by default: all 
    sent_idx = get_sent_idx(df, candidate_idx, sample_ratio, remove_noconstr_sent=True)  
    print("%s sentences selected" % len(sent_idx))
    # print()
    # sent_set = prepare_sent(df, sent_idx, col='Normalized_descp', lower=True)
    dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    # no need to encode word

    # iterate all the sentences twice
    # iter1: for each sentence 
    curr_idx = 0
    for sidx in sent_idx:
        sent = prepare_sent(df, sidx, col='Normalized_descp', lower=True)
        # print("%s \t %s" %(sidx, sent))
        #  get dependency parsing tree
        _, horizontal_format = generate_parsing_tree(dependency_parser, sent)
        # get all subtrees 
        # decoded_all_subtree: list of strs
        decoded_all_subtree, word_map, word_map_inverse = get_all_subtree(sent, horizontal_format, max_iter, threadsafe=True)
        #  put data into `subtree_map`, `reverse_subtree_map`, `subtree_app`
        for subtree in decoded_all_subtree:
            if subtree in reverse_subtree_map:
                subtree_idx = reverse_subtree_map[subtree]
            else:
                subtree_idx = curr_idx
                curr_idx += 1
                subtree_map[subtree_idx] = subtree
                reverse_subtree_map[subtree] = subtree_idx
                # calculate `subtree_size`
                subtree_size[subtree_idx] = get_tree_size(subtree)
            subtree_app[subtree_idx].append(sidx)
    
    print("%s subtrees from %s sentences" % (len(list(subtree_map.keys())), len(sent_idx)))
    # iter2: for each sentence
    curr_idx = 0
    for sidx in sent_idx:
        row = df.iloc[sidx]
        all_ir = parse_ir(row, constr_cols)   # dict: category -> set of IRs
        assert not constr_empty(all_ir)
        # print(all_ir)
        for cat in all_ir:
            # for each IR
            for ir in all_ir[cat]:
            #   put data into `ir_map`, `reverse_ir_map`, `ir_app`
                if ir in reverse_ir_map[cat]:
                    ir_idx = reverse_ir_map[cat][ir]
                else:
                    ir_idx = curr_idx
                    curr_idx += 1
                    ir_map[ir_idx] = (cat, ir)
                    reverse_ir_map[cat][ir] = ir_idx
                ir_app[ir_idx].append(sidx)

    print("%s IRs from %s sentences" % (len(list(ir_map.keys())), len(sent_idx)))
    # check the idx is continuous
    check_key_cont(ir_map)
    check_key_cont(subtree_map)

    # check app non-empty
    check_app_non_empty(subtree_app)
    check_app_non_empty(ir_map)
    print('Finished checking')
    # get the value of m,n 
    m = len(list(ir_map.keys()))
    n = len(list(subtree_map.keys()))
    # init `cooccur_cnt` (n*m)
    cooccur_cnt = [[0 for j in range(m)] for i in range(n)]
    prob = [[0 for j in range(m)] for i in range(n)]


    # update values in `cooccur_cnt` by calculating the overlap between `subtree_app` and `ir_app`
    for i in range(n): # subtree idx
        for j in range(m):  # ir idx
            cooccur_cnt[i][j] = count_overlap(subtree_app[i], ir_app[j])
            # calculate `prob` with `cooccur_cnt` and `subtree_app`
            prob[i][j] = cooccur_cnt[i][j]/len(subtree_app[i])

    # save data
    dump_pickle(os.path.join(save_path, 'subtree_map'), subtree_map)
    dump_pickle(os.path.join(save_path, 'reverse_subtree_map'), reverse_subtree_map)
    dump_pickle(os.path.join(save_path, 'ir_map'), ir_map)
    dump_pickle(os.path.join(save_path, 'reverse_ir_map'), reverse_ir_map)

    dump_pickle(os.path.join(save_path, 'subtree_app'), subtree_app)
    dump_pickle(os.path.join(save_path, 'ir_app'), ir_app)

    dump_pickle(os.path.join(save_path, 'subtree_size'), subtree_size)
    dump_pickle(os.path.join(save_path, 'cooccur_cnt'), cooccur_cnt)
    dump_pickle(os.path.join(save_path, 'prob'), prob)



    # save_yaml(os.path.join(save_path, 'subtree_map'), subtree_map)
    # save_yaml(os.path.join(save_path, 'reverse_subtree_map'), reverse_subtree_map)
    # save_yaml(os.path.join(save_path, 'ir_map'), ir_map)
    # save_yaml(os.path.join(save_path, 'reverse_ir_map'), reverse_ir_map)

    # save_yaml(os.path.join(save_path, 'subtree_app'), subtree_app)
    # save_yaml(os.path.join(save_path, 'ir_app'), ir_app)

    # save_yaml(os.path.join(save_path, 'subtree_size'), subtree_size)
    # save_yaml(os.path.join(save_path, 'cooccur_cnt'), cooccur_cnt)
    # save_yaml(os.path.join(save_path, 'prob'), prob)




    


run('./sample/tf30_merged.csv', './data/tf/', max_iter=7, sample_ratio=0.1)
# run('./sample/pt30_merged.csv', './data/pt/', max_iter=-1)
# run('./sample/mx30_merged.csv', './data/mx/', max_iter=-1)



# tf 0.01
# 15 sentences selected
# 89895 subtrees from 15 sentences
# 9 IRs from 15 sentences
