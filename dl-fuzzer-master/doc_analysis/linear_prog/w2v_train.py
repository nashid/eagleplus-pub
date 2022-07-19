from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from util import *
import pickle
import yaml
import os

def load_data(folder_path):
    data = []
    yaml_files = get_file_list(folder_path)
    for f in yaml_files:
        parser = read_yaml(os.path.join(folder_path, f))
        for arg in (parser['constraints']):
            data += parser['constraints'][arg].get('normalized_descp', [])
            if 'normalized_default' in parser['constraints'][arg]:
                data.append(parser['constraints'][arg]['normalized_default'])
            if 'normalized_docdtype' in parser['constraints'][arg]:
                data.append(parser['constraints'][arg]['normalized_docdtype'])
    return data
                
            
sentences = []
sentences = load_data('./normalized_doc_icse/tf/')
sentences += load_data('./normalized_doc_icse/pt/')
sentences += load_data('./normalized_doc_icse/mx/')

# hardcode
extra_word = ['dtype', 'structure', 'shape', 'ndim', 'enum', 'int', 'float', 'string', 'numeric', 'bool', 'tuple']

for w in extra_word:
    sentences.append(w)

for i in range(5):
    sentences.append(str(i))

# sentences_with_label = []

# for label in labels:
#     for sentence in sentences:
#         sentences_with_label.append(sentence + ' ' + label)

# to convert from ['hello world'] to [['hello', 'world']]
p_sentences = [sen.split() for sen in sentences]


model = Word2Vec(sentences=p_sentences, window=5, min_count=1, workers=1)

model.train(p_sentences, total_examples=1, epochs=100)

model.save('w2v_data/w2v.model')

dump_pickle('w2v_data/w2v_train_data', sentences)
# file = open('w2v_data/w2v_train_data', 'wb')
# pickle.dump(sentences, file)
# file.close()