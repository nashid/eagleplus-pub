import numpy as np
import torch
import os
from datetime import datetime
import argparse
import traceback
import pickle
import importlib
import random
from timeit import default_timer as timer

from rules.equiv_util import get_func_ptr, main, load_argument_file, load_input_file, get_log_file, save_output_data


def test_rule_cpuvsgpu(input, target_fun, log_file):
    seed = 0
    # fix seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # run function on cpu
    try:
        output_1 = target_fun(**input)
        # print(input_list)
        output_1_numpy = output_1.cpu().detach().numpy()
    except:
        with open(log_file, "a+") as f:
            f.write(traceback.format_exc())
        # print(traceback.format_exc())
        output_1_numpy = None

    # move tensors to gpu
    for key in input.keys():
        if isinstance(input[key], torch.Tensor):
            input[key] = input[key].cuda()

    # reset seed 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # run function on gpu
    try:
        output_2 = target_fun(**input)
        # print(type2_input_list)
        output_2_numpy = output_2.cpu().detach().numpy()
    except:
        with open(log_file, "a+") as f:
            f.write(traceback.format_exc())
        # print(traceback.format_exc())
        output_2_numpy = None

    return [output_1_numpy, output_2_numpy]


def run(in_dir, out_dir, lib, version, api_config, input_index):

    api_name = api_config

    argument = load_argument_file(in_dir, lib, version, api_name, input_index)
    log_file = get_log_file(out_dir, lib, version, 'rule_17', api_config, input_index)

    # get function pointer
    package = "torch"
    mod = importlib.import_module(package)
    target_fun = get_func_ptr(mod, api_name)

    # run test
    [output_1, output_2] = test_rule_cpuvsgpu(argument, target_fun, log_file)

    save_output_data([output_1, output_2], out_dir, lib, version, 'rule_17', api_config, input_index)

    return output_1 is not None and output_2 is not None


if __name__ == "__main__":
    main(run)
