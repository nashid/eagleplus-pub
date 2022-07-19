# EAGLE - NEW RULES (Distributed vs Non-distributed)

The default tested PyTorch version is 1.22. 

## Folder/File Structure

Rule 17: DistributedModelParallel inference

Rule 18: DistributedModelParallel training

Rule 19: DistributedDataParallel training

File `rule_17_pytorch_scripts.py` is the core file of chekcing the consistency of DistributedModelParallel inference. 

File `gen_torchrec_model_and_dataset.py` generate models to test the new rules. 

Files `torchrec_distributed.py` and `trochrec_tutorial.py` were copied and modified based on  torchrec files. They are not part of the experiments. 

## Instruction

### Create enviroment

### Generate models
`cd EAGLE`

`python -m newrules.gen_torchrec_model_and_dataset`

### Execute EAGLE rules
To execute rule 17, execute  
`cd EAGLE`

`bash execute_testing_new_rules.sh`. 

### Analyze results
To analyze rule 17's results, 
`cd EAGLE`

`python analyze_results_distributed.py`

