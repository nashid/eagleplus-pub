import os
import yaml

def save_file(fpath, content):
    with open(fpath, 'w+') as wf:
        if isinstance(content, str):
            wf.write(content)
        elif isinstance(content, list):
            wf.write('\n'.join(content))

def read_yaml(path):
    with open(path) as yml_file:
        ret = yaml.load(yml_file, Loader=yaml.FullLoader)
    return ret
    
def get_file_list(dir_addr):

    files = []
    for _,_, filenames in os.walk(dir_addr):
        files.extend(filenames)
        break

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    return files


#folder = '/home/danning/Desktop/DocTer/dl-fuzzer/doc_analysis/'
path_raw = {
    'collect_doc/tf/tf21_all_src': 'tf2.1',
    'collect_doc/tf/tfdoc2.2': 'tf2.2',
    'collect_doc/tf/tfdoc2.3': 'tf2.3',
    'collect_doc/pytorch/pt1.4_parsed': 'pt1.4',
    'collect_doc/pytorch/pt15_all_src': 'pt1.5',
    'collect_doc/pytorch/pt1.6_parsed': 'pt1.6',
    'collect_doc/pytorch/pt1.7_parsed': 'pt1.7'

}

path_constr = {
    '../doc_analysis/extract_constraint/tf/tf21_all/changed/': 'tf2.1',
    '../doc_analysis/extract_constraint/tf/tf22/changed': 'tf2.2',
    '../doc_analysis/extract_constraint/tf/tf23/changed': 'tf2.3',
    '../doc_analysis/extract_constraint/pytorch/pt14/changed': 'pt1.4',
    '../doc_analysis/extract_constraint/pytorch/pt15_all': 'pt1.5',
    '../doc_analysis/extract_constraint/pytorch/pt16': 'pt1.6',
    '../doc_analysis/extract_constraint/pytorch/pt17': 'pt1.7',

}

# for p in path:
#     flist = get_file_list(os.path.join(folder, p))
#     flist = [x[:-5] for x in flist]
#     save_file(path[p], flist)


for p in path_constr:
    # print(os.path.join(folder, p))
    # print(folder)
    # print(p)
    flist = get_file_list(p)
    flist_complex = []
    flist_sparse = []
    for f in flist:
        content = read_yaml(os.path.join(p, f))
        for arg in content['constraints']:
            if 'complex' in str(content['constraints'][arg].get('dtype', [])).lower():
                flist_complex.append(f[:-5]+'-'+arg)
            if 'sparse' in str(content['constraints'][arg].get('tensor_t', [])).lower():
                flist_sparse.append(f[:-5]+'-'+arg)
    save_file('./{}_complex'.format(path_constr[p]), flist_complex)
    save_file('./{}_sparse'.format(path_constr[p]), flist_sparse)

