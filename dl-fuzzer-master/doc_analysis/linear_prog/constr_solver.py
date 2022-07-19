import collections
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import copy
from util import *
import os 

os.environ['GRB_LICENSE_FILE'] = "/Users/danning/license/gurobi.lic"


# run command: grbgetkey f73732c6-4d8d-11ec-af9b-0242ac150003
# export GRB_LICENSE_FILE="/Users/danning/license"


constr_cat = {
    'dtype': 'dtype', 
    'structure': 'structure', 
    'tensor_t': 'structure',    # merge into structure 
    'shape': 'shape', 
    'ndim': 'ndim', 
    'range' : 'range',
    'enum': 'enum'
    }
    


def optimize(data_path, weight, priority): 
    subtree_size = load_pickle(os.path.join(data_path, 'subtree_size'))   # length=n
    prob = load_pickle(os.path.join(data_path, 'prob'))   # n*m
    n = len(prob)       # number of subtree
    m = len(prob[0])    # number of IR
    prob_sum = sum([sum(prob[i]) for i in range(n)]) # trivial result when all x are 1
    size_sum = sum(list(subtree_size.values()))        # trivial result when all x are 1
    print(prob_sum)         # up limit of term1
    print(m)                # up limit of term2
    print(n)                # up limit of term3
    print(size_sum)         # up limit of term4
    # return 

    model = gp.Model()
    # get license: https://www.gurobi.com/academia/academic-program-and-licenses/
    x = model.addVars(n,m, vtype=GRB.BINARY)
    c = model.addVars(m, vtype=GRB.BINARY)
    d = model.addVars(n, vtype=GRB.BINARY)
    # return
    # add contraints to c
    for j in range(m):
        model.addGenConstrIndicator(c[j], True, x.sum('*', j), GRB.GREATER_EQUAL, 1.0)
    for j in range(m):
        model.addGenConstrIndicator(c[j], False, x.sum('*', j), GRB.EQUAL, 0.0)

    # add contraints to d
    for i in range(n):
        model.addGenConstrIndicator(d[i], True, x.sum(i, '*'), GRB.GREATER_EQUAL, 1.0)
    for i in range(n):
        model.addGenConstrIndicator(d[i], False, x.sum(i, '*'), GRB.EQUAL, 0.0)

    # term1: maximize the sum
    model.setObjectiveN(-sum(x[i,j]*prob[i][j] for i in range(n) for j in range(m)), index=0, weight=weight[0], priority=priority[0], name='term1')

    # term2: maximize sum of c
    model.setObjectiveN(-c.sum(), index=1, weight=weight[1], priority=priority[1], name='term2')

    # term3: minimize sum of d
    model.setObjectiveN(d.sum(), index=2, weight=weight[2], priority=priority[2], name='term3')

    # term4: minimize the size of individual tree
    model.setObjectiveN(sum(d[i]*subtree_size[i] for i in range(n)), index=3, weight=weight[3], priority=priority[3], name='term4')


    model.optimize()
    for i in range(model.NumObj):
        model.setParam(gp.GRB.Param.ObjNumber, i)
        print(f"Obj {i+1} = {model.ObjNVal}")

    return model.getAttr('x', x), n, m

def merge_constr(constr1, constr2):
    new_constr = copy.deepcopy(constr1)

    for subcat in constr2:
        cat = constr_cat[subcat]
        if cat in new_constr:
            new_constr[cat] = list(set(new_constr[cat]+constr2[cat]))
        else:
            new_constr[cat] = constr2[cat]

    return new_constr


def gen_rule(data_path, x, n, m):
    subtree_map = load_pickle(os.path.join(data_path, 'subtree_map'))
    ir_map = load_pickle(os.path.join(data_path, 'ir_map'))
    rule = collections.defaultdict(dict)
    for i in range(n):
        for j in range(m):
            if x[(i,j)] == 1:
                tree = subtree_map[i]
                subcat, ir_val = ir_map[j]
                rule[tree] = merge_constr(rule[tree], {subcat:[ir_val]})
            else:
                pass
                # print('not selected')
                
    return rule


def main(data_path, weight=[2,1,1,1], priority=[1,1,1,1]):
    x, n, m = optimize(data_path, weight, priority)
    rule = gen_rule(data_path, x, n, m)
    print('Extract %s rules from %s subtree and %s IR' % (len(list(rule.keys())), n, m))
    save_yaml(os.path.join(data_path, 'subtree_rule.yaml'), rule)


main('./data/tf/')
# main('./data/pt/')
# main('./data/mx/')