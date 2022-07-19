from gensim.test.utils import common_texts
from gensim.models import Word2Vec
#from gensim.models import Doc2Vec
from util import *
from mining_util import *
from scipy import spatial
import pickle


# import yaml
import os
import pandas as pd




#model.wv.get_vector("D")

# descp = "A CONSTANT_NUM D BSTR D_STRUCTURE with the same shape as QSTR"
# label = ['dtype', 'structure', 'shape', 'ndim', 'enum']
# descp_list = descp.split( )
# print(descp_list)
# sen_vec = [0] * 100
# for i in descp_list:
#     sen_vec += model.wv.get_vector(i)

# for i in label:
#     w_vec = model.wv.get_vector(i)
#     sim = 1 - spatial.distance.cosine(sen_vec, w_vec)
#     print(str(i) + ":" + str(sim))


constr_cols = ['dtype', 'structure', 'shape', 'ndim', 'enum', 'range']



def get_vec(model, str_list):
    # input: a list of strings(words) or a sentence (string)
    if isinstance(str_list, str):
        str_list = str_list.split()
    assert isinstance(str_list, list)
    sen_vec = [0] * 100

    for w in str_list:
        sen_vec += model.wv.get_vector(w)
    return sen_vec
    


def cal_dist(vec1, vec2):
    # TODO: check other dist, e.g., Word Mover's Distance
    dist = 1 - spatial.distance.cosine(vec1, vec2)
    return dist
    
def normalize_ir(ir):
    ir = re.sub('[\[\(]0,inf[\)\]]', 'positive', ir)
    ir = re.sub('[\[\(]inf,0[\)\]]', 'negative', ir)
    ir = re.sub('tf.dtype', 'dtype', ir)
    ir = re.sub('numpy.dtype', 'dtype', ir)
    ir = re.sub('torch.dtype', 'dtype', ir)
    # ir = re.sub('[^0-9a-zA-Z_]+', '', ir)
    ir = re.sub('&', '', ir)
    ir = re.sub('[\[\]\(\)]', '', ir)       #shape [constant_num] -> constant_num
    return ir 

def test_on_csv(csv_path, model_path, save_path):
    df = pd.read_csv(csv_path)     
    model = Word2Vec.load(model_path)
    dist_list = []  
    errors = []
    for index, row in df.iterrows(): 
        if row.isnull()['Normalized_descp']:
            continue
        # get all constr
        try:
            all_ir = parse_ir(row, constr_cols) 
            if constr_empty(all_ir):   # if this row has no IRs
                continue
            sen_vec = get_vec(model, row['Normalized_descp'])
            for cat in all_ir:  
                for ir in all_ir[cat]:  # skip if empty
                    
                    ir_vec = get_vec(model, [cat, normalize_ir(ir)])

                    dist_list.append(cal_dist(sen_vec, ir_vec))
        except Exception as e: 
            # print(row)
            # print(row['Normalized_descp'])
            # print([cat, normalize_ir(ir)])
            print(e)
            errors.append(e)
            # print()
            # break
    dump_pickle(save_path, dist_list)
    # save_yaml(os.path.join(save_path, file_name), errors)
    # file = open(save_path, 'wb')
    # pickle.dump(dist_list, file)


            


# test_on_csv('sample/tf30_merged.csv', 'w2v_data/w2v.model', 'w2v_data/tf_label_sim')
# test_on_csv('sample/pt30_merged.csv', 'w2v_data/w2v.model', 'w2v_data/pt_label_sim')
test_on_csv('sample/mx30_merged.csv', 'w2v_data/w2v.model', 'w2v_data/mx_label_sim')

