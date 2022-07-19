# analyze the results for rule 17 
import os
import pickle
from torchsnapshot import Snapshot
import torch

from newrules.rule_17_pytorch_script import run, compare_result
from newrules.torchrec_benchmark import gen_ebc_comparison_dlrm, gen_fused_ebc_uvm, gen_ebc_comparison_scaling, get_ebc_fused_ebc_model, get_fused_ebc_uvm_model, get_random_dataset

def calculate_max_error(result_list):
    match = True
    max_diff = 0
    # print(result_list[0])
    if result_list[0] is None:
        match = False
        # print("result_list[0] is None")
    else:
        if type(result_list[0]) == torch.Tensor:
            for i in range(1, len(result_list)):
                if result_list[i] is None:
                    match = False
                    continue
                
                diff = torch.linalg.norm(result_list[0] - result_list[i])
                # print(diff)
                if diff > max_diff:
                    max_diff = diff

                if not torch.allclose(result_list[0], result_list[i]):
                    # print("Error: result mismatch, 0 vs {}".format(i))
                    match = False

        else: # if the output is dictionary
            # for results of different world_size or nondistributed
            for i in range(1, len(result_list)):
                if result_list[i] is None:
                    match = False
                    continue
                if result_list[0].keys() != result_list[i].keys():
                    # print("Error: result key mismatch, 0 vs {}".format(i))
                    match = False
                    continue
                    
                for key, value in result_list[0].items():
                    # print((value - result_list[i][key]).view(-1).shape)
                    diff = torch.linalg.norm((value - result_list[i][key]))
                    
                    # print(diff)
                    if diff > max_diff:
                        max_diff = diff
                    if not torch.allclose(value, result_list[i][key]):
                        # print("Error: result value mismatch, 0 vs {}".format(i))
                        match = False
                        
    return match, max_diff

def main():
    model_saving_root = "./data"

    # device = torch.device("meta")

    # backend = "gloo"
    # backend = "nccl"

    result_dir_root="data/outputs"
    if not os.path.exists(result_dir_root):
        os.makedirs(result_dir_root)

    model_config_list_1 = []  # feed to get_ebc_fused_ebc_model
    model_config_list_2 = []  # feed to get_fused_ebc_uvm_model

    model_config_list_1.extend(gen_ebc_comparison_dlrm())
    model_config_list_1.extend(gen_ebc_comparison_scaling())
    model_config_list_2.extend(gen_fused_ebc_uvm())


    # print(len(model_config_list_1), len(model_config_list_2))

    for i in range(len(model_config_list_1)):
        if i >= 125: #TODO: keep this the same as the number of models generated 
            break

        with open(os.path.join(model_saving_root, "outputs", "result_1_ebc_%d" % i), "rb") as f:
            result_ebc_list = pickle.load(f)
        match, diff = calculate_max_error(result_ebc_list)
        if not match:
            print("Error: result_1_ebc_{}, diff: {}".format(i, diff))
        else:
            print("Success: result_1_ebc_{}, diff: {}".format(i, diff))
        
        with open(os.path.join(model_saving_root, "outputs", "result_1_fused_ebc_%d" % i), "rb") as f:
            result_fused_ebc_list = pickle.load(f)
        match, diff = calculate_max_error(result_fused_ebc_list)
        if not match:
            print("Error: result_1_ebc_{}, diff: {}".format(i, diff))
        else:
            print("Success: result_1_fused_ebc_{}, diff: {}".format(i, diff))
        
        # print(result_ebc_list, result_fused_ebc_list)
    
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
    main()