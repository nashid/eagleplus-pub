import os
import pickle
from torchsnapshot import Snapshot
import torch
import sys, getopt # for parsing command arguments

from newrules.rule_17_pytorch_script import run
from newrules.torchrec_benchmark import gen_ebc_comparison_dlrm, gen_fused_ebc_uvm, gen_ebc_comparison_scaling, get_ebc_fused_ebc_model, get_fused_ebc_uvm_model, get_random_dataset

def main(argv):

    # the maximum number of model that we can support
    MAX_MODEL = 500;

    model_saving_root = "./data" #TODO: set this as a parameter

    device = torch.device("meta")

    backend = "gloo"
    # backend = "nccl"

    result_dir_root="data/outputs"
    if not os.path.exists(result_dir_root):
        os.makedirs(result_dir_root)

    model_config_list_1 = []  # feed to get_ebc_fused_ebc_model
    model_config_list_2 = []  # feed to get_fused_ebc_uvm_model

    model_config_list_1.extend(gen_ebc_comparison_dlrm())
    model_config_list_1.extend(gen_ebc_comparison_scaling())
    model_config_list_2.extend(gen_fused_ebc_uvm())


    print(len(model_config_list_1), len(model_config_list_2))

    #process command line arguments
    if len(sys.argv) < 2:
        print('Using default model number (using 1 model) to test.')
        modelnum = 1
    else: 
        modelnum = int(sys.argv[1])
        if modelnum < 1 or modelnum > MAX_MODEL:
            print('Model number ' + str(modelnum) + ' is out of range. Should be between 1 and ' + str(MAX_MODEL)+'.')
            return
        print('Using the entered modelnum ' + str(modelnum) +' to test this rule.')

    # control the list of models to test this rule
    for i in range(len(model_config_list_1)):
        if i >= modelnum: #170: # if i > 0 or i>=1 , then it is only testing with one model DONE: turn this into a parameter
            break
        config = model_config_list_1[i]
        # config = config[0:1]

        #initialize the model
        model, model_fuse = get_ebc_fused_ebc_model(config, device)

        app_state = {"model": model}
        model_saving_path = os.path.join(model_saving_root, "models", "model_1_ebc_%d" % i)

        snapshot = Snapshot(path=model_saving_path)

        # load the generated models
        snapshot.restore(app_state)

        app_state = {"model": model_fuse}
        model_saving_path = os.path.join(model_saving_root, "models", "model_1_fused_ebc_%d" % i)
        snapshot = Snapshot(path=model_saving_path)
        snapshot.restore(app_state)

        with open(os.path.join(model_saving_root, "dataset", "dataset_1_%d" % i), "rb") as f:
            example = pickle.load(f)
        
        # print(model(example.sparse_features).to_dict())
        # print(model_fuse(example.sparse_features).to_dict())
        # break

        test_x = [example.sparse_features]
        result_file_path = os.path.join(result_dir_root, "result_1_ebc_%d" % i)
        log_file_path = os.path.join(result_dir_root, "log_1_ebc_%d" % i)
        run(model, test_x, sharding_type="TABLE_WISE", backend=backend, world_size=2, result_dir=result_file_path, tmp_dir="./", log_file=log_file_path)

        result_file_path = os.path.join(result_dir_root, "result_1_fused_ebc_%d" % i)
        log_file_path = os.path.join(result_dir_root, "log_1_fused_ebc_%d" % i)
        run(model_fuse, test_x, sharding_type="TABLE_WISE", backend=backend, world_size=2, result_dir=result_file_path, tmp_dir="./", log_file=log_file_path)
        # return
    
    return

    for i in range(len(model_config_list_2)):
        config, location = model_config_list_2[i]

        model = get_fused_ebc_uvm_model(config, device, location)

        app_state = {"model": model}
        model_saving_path = os.path.join(model_saving_root, "models", "model_2_%d" % i)
        snapshot = Snapshot(path=model_saving_path)
        snapshot.restore(app_state)

        with open(os.path.join(model_saving_root, "dataset", "dataset_2_%d" % i), "rb") as f:
            example = pickle.load(f)
        
        print(model(example.sparse_features))

        test_x = [example.sparse_features]
        result_file_path = os.path.join(result_dir_root, "result_2_%d" % i)
        log_file_path = os.path.join(result_dir_root, "log_2%d" % i)
        run(model, test_x, sharding_type="TABLE_WISE", backend=backend, world_size=2, result_dir=result_file_path, tmp_dir="./", log_file="./log.txt")

    return


if __name__ == "__main__":
    main(sys.argv[1:])
