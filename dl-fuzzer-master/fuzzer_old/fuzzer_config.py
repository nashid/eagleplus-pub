import os

import networkx as nx

import util
from param import Param


class FuzzerConfig:
    def __init__(self, kwargs):
        allowed_kwargs = {
            'adapt_to',
            'cluster',  # not used
            'cluster_only',
            'consec_fail',
            'data_construct',
            'dist_metric',  # not used
            'dist_threshold',  # not used
            'dtype_config',
            'fuzz_optional',
            'gen_script',
            'ignore',
            'max_iter',  # not used
            'max_time',  # not used
            'obey',  # not used
            'target_config',
            'timeout',
            'verbose',
            'workdir',  # not used
        }

        util.validate_kwargs(kwargs, allowed_kwargs)

        # user supplied configs
        self.adapt_to = kwargs.get('adapt_to')
        self.consec_fail = kwargs.get('consec_fail')
        self.fuzz_optional = kwargs.get('fuzz_optional')
        self.ignore_constraint = kwargs.get('ignore')
        self.max_iter = kwargs.get('max_iter')
        self.max_time = kwargs.get('max_time')
        self.obey = kwargs.get('obey')
        self.timeout = kwargs.get('timeout')
        self.verbose = kwargs.get('verbose')
        self.workdir = kwargs.get('workdir', os.getcwd())

        # dtype config
        dtype_config = kwargs.get('dtype_config')
        if dtype_config is None:
            util.error('missing required dtype config.')
        self.dtype_map = dtype_config

        # target config
        target_config = kwargs.get('target_config')
        if target_config is None:
            util.error('missing required target config.')
        self.package = target_config.get('package')
        self.version = target_config.get('version')
        self.target = target_config.get('target')
        self.title = target_config.get('title')
        exceptions = target_config.get('exceptions', [])
        if isinstance(exceptions, list):
            exceptions = [list(dict(e).keys())[0] for e in exceptions]
        else:
            exceptions = []

        self.exceptions = exceptions

        # constraints from target config
        constraints = target_config.get('constraints') if not self.ignore_constraint else None

        def get_input_names(inputs, key):
            if inputs is None:
                util.error('no "inputs" in the target config')
            return [p for p in inputs.get(key, []) if p not in ['**kwargs', '**args']]

        required_inputs = get_input_names(target_config.get('inputs'), 'required')
        optional_inputs = get_input_names(target_config.get('inputs'), 'optional')

        # create param obj from inputs
        def create_param(inputs, is_required):
            param_dict = {}
            for one_input in inputs:
                input_constrt = self.check_one_input_constrt(constraints, one_input)
                one_param = Param(one_input, input_constrt, self.dtype_map, required=is_required)
                param_dict[one_input] = one_param
            return param_dict

        self.required_param = create_param(required_inputs, True)
        self.optional_param = create_param(optional_inputs, False)

        self.gen_order = self.generation_order()
        self.check_config_validity()

    def check_one_input_constrt(self, constraints, input_name):
        if constraints is None:
            return None
        input_constrt = constraints.get(input_name)
        # special check to confirm there exist usable constraints
        constrt_cats = ['dtype', 'ndim', 'shape', 'range', 'enum', 'tensor_t', 'structure', 'default']
        if any(k in constrt_cats for k in input_constrt.keys()):
            return input_constrt
        else:  # no useful constraints, simply set it to None
            return None

    def generation_order(self):
        DG = nx.DiGraph()
        params = list(self.required_param.keys()) + list(self.optional_param.keys())
        DG.add_nodes_from(params)

        def add_edge(param_dict):
            for pname, p_obj in param_dict.items():
                for dname in p_obj.var_dep:
                    assert dname != ''
                    if dname == pname:
                        util.error('cannot refer to the parameter itself as a dependency.')
                    # note: it's impossible dname is in both required_param and optional param
                    if dname in self.required_param and dname in self.optional_param:
                        util.error('"%s" cannot be in both required and optional input' % dname)
                    if dname not in self.required_param and dname not in self.optional_param:
                        util.error('"%s" does not exist in "inputs" of target config' % dname)

                    d_obj = self.required_param.get(dname) or self.optional_param.get(dname)
                    if d_obj is None:
                        util.error('dependent "%s" is not in the parameter list' % dname)
                    d_obj.mark_shape_dep()
                    p_obj.add_dep_param_obj(d_obj)
                    DG.add_edge(dname, pname)

        add_edge(self.required_param)
        add_edge(self.optional_param)
        try:
            return list(nx.algorithms.dag.topological_sort(DG))
        except nx.exception.NetworkXUnfeasible:
            util.error('parameter dependencies seem to contain a cycle; consider to fix the yaml spec.')

    def check_config_validity(self):
        for attr, value in self.__dict__.items():
            if value is None:
                util.warning('"%s" is None' % attr)
        return
