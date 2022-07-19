from pandas.core.indexes.timedeltas import timedelta_range
import yaml
import pandas as pd
import os

def save_yaml(path, data):
    with open(path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

def read_yaml(path):
    with open(path) as yml_file:
        ret = yaml.load(yml_file, Loader=yaml.FullLoader)
    return ret

def init_yaml(api_name):
    ret = {}
    ret['title'] = api_name
    ret['target'] = api_name.split('.')[-1]
    ret['package'] = 'tensorflow'
    ret['version'] = '2.1.0'
    ret['inputs'] = {}
    ret['inputs']['optional'] = []
    ret['inputs']['required'] = []
    ret['constraints'] = {}
    return ret
    
def gen_arg_data(argname, row):
    ret = {}
    dep = []
    all_cat = ['dtype', 'structure', 'tensor_t', 'shape', 'ndim', 'range', 'enum', 'default']
    for cat in all_cat:
        if not pd.isna(row[cat]):
            if cat == 'ndim':
                ret[cat] = [str(int(row[cat]))]
            else:
                ret[cat] = [row[cat]]
        if cat == 'shape' and row[cat] == "[a,b,&input_size]":
            dep = ['a','b']
    return ret, dep

def main(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    all_data = {}
    for index, row in df.iterrows():
        api_name = row['API']
        if api_name not in all_data:
            all_data[api_name] = init_yaml(api_name)
        argname = row['argname']
        if row['required'] == 'Y':
            all_data[api_name]['inputs']['required'].append(argname)
        elif row['required'] == 'N':
            all_data[api_name]['inputs']['optional'].append(argname)
        else:
            print(row['required'])
            print('Error')
            continue
        
        argdata, dep = gen_arg_data(argname, row)
        all_data[api_name]['constraints'][argname] = argdata
        if dep:
            all_data[api_name]['dependency'] = all_data[api_name].get('dependency', []) + dep


        
        
    for api in all_data:
        if all_data[api].get('dependency', []):
            all_data[api]['dependency'] = list(set(all_data[api]['dependency']))
        
        fname = api.lower() + '.yaml'
        save_yaml(os.path.join(save_dir, fname), all_data[api])

main('./torchvision3.csv', './torchvision_constr3/')
    


