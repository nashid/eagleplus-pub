Before running, don't forget to update `$home` directory in `run_batch.sh`


### folders:
$home
    |___ $home/code/DocTer_Ext
    |___ $home/workdir


### Command for experiment

Note that `fuzz_optional` is diabled


- baseline: `bash run_batch.sh .../changed expect_ok prev_ok tensorflow no_constr no_adapt`
- adt_nc (baseline+adpt):  `bash run_batch.sh .../changed expect_ok prev_ok tensorflow no_constr adapt`
- eo_nat: `bash run_batch.sh .../changed expect_ok prev_ok tensorflow constr no_adapt`
- ee: `bash run_batch.sh .../changed expect_exception prev_ok tensorflow no_constr no_adapt`

Current dir to yaml files: 
`/Users/danning/Desktop/deepflaw/exp/code/DocTer-Ext/doc_analysis/extract_constraint/tf/constraint_layer/changed`