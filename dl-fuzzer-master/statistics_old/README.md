# Result statistics for our fuzzer
This directory contains scripts for running the fuzzzer in batch, collecting statistics for generated inputs, and inspecting them.

## Important files

* ```run_docker_setup.sh``` : This file creates the given number of docker containers and runs our fuzzer in each container. The insructions are written below. 
* ```collect_error_msgs.py``` : This file outputs a csv file that contains statistics and error messages of generated tests in the given workdir. (usage example: ```python collect_error_msgs.py --workdir=/.../docker_9_e5c96e_no_pp/workdir/expect_ok --output=/absolute_path/.../docker9_no_adapt.csv```) 
* ```prepare_bug_list.sh``` : This file prints in the csv format the list of severe bugs (e.g., segfault, floating point) in the workdir. (usage example: ```bash prepare_bug_list.sh /.../docker_9_e5c96e_no_pp/workdir expect_ok false constr no_adapt```)  

## How to use ```run_docker_setup.sh```
This script automatically creates the given number of docker containers and runs ```divide_constraints_dir.sh``` and ```run_batch.sh``` inside the created docker containers. 

**Step 0**

* Create a docker home directory, say ```docker_home```. e.g., /path/to/expr/tensorflow/docker_XXXX.
* Copy the latest ```dl-fuzzer``` repository under ```docker_home/code```. This will create ```docker_home/code/dl-fuzzer```

**Step 1**

Go to ```docker_home/code/dl-fuzzer/statistics```. And, edit ```run_docker_setup.sh``` to specify the fuzzer and docker options defined in variables below. The variables expect specific values. 

* ```$expects```: a list that contains one of [expect_ok | expect_exceptions].
* ```$modes```: a list that contains one of [prev_ok | permute].
* ```$constr```: one of [contr | no_constr].
* ```$adaptive_gen```: one of [adapt | no_adapt].

**Step 2**

Run ```run_docker_setup.sh``` that requires 4 arguments below.

* ```$package```: one of [tensorflow | pytorch | mxnet].
* ```$division_count```: the number of docker containers to create.
* ```$docker_home```: the absolute path to the docker home directory to be mounted.
* ```$docker_surfix```: the surfix used to name the docker containers.

**Usage example**
```
bash run_docker_setup.sh tensorflow 10 docker_home tf
```

**Important note**

This script runs the commands in ```screen``` inside the docker container. Once the ```screen``` is created in the docker, you need to detach it manually by entering ```ctrl-a``` and ```d``` to get out of the ```screen```. If you don't want to use ```screen```, you can modify the script accordingly. 



## ~~How to run ```run_batch.sh```~~ (deprecated)
This script runs the fuzzer for all ```yaml``` files in the given directory. A ```yaml``` file is a configuration file that contains a set of constraints of an API under test (e.g., ```add.yaml``` contains constraints of an API named ```add```).

**A usuage example below. Note that you need to set up your own ```home``` directory in the file before use.**

```
bash run_batch.sh /absolute path/to/dl-fuzzer/constraints/constraints_1/changed True prev_ok
```

* Argument 1: The **absolute** path to the directory that contains ```yaml``` files.

* Argument 2: A ```boolean``` flag whether to run the fuzzer in ```expect_ok``` mode. For ```expect_ok```, give ```True```, for ```expect_exception```, give ```False```. 

* Argument 3: The type of fuzzing algorithms. Currently, we support ```prev_ok``` and ```permute```

## ~~How to run ```run_batch_stat.py```~~ (deprecated: now refer to ```collect_error_msgs.py```)
This script creates a ```csv``` file that lists statistics of generated inputs (e.g., # of generated inputs, # of failures, etc.)  for the tested APIs.

**A usuage example below.**

```
python run_batch_stat.py --workdir=/path/to/.workdir/expect_ok_prev_ok --yamldir=/path/to/dl-fuzzer/constraints/constraints_1/changed --output=results/fuzzing_stats.csv
```

* Argument 1: The path to the parent directory that contains ```XXX.yaml_workdir``` directories.

* Argument 2 (optional): The path to the directory that contains ```yaml``` files to check if the generated inputs are for the valid ```yaml``` files. If not provided, it uses ```../constraints/constraints_1/changed ``` as default. 

* Argument 3 (optional): The name of output csv file. If not provided, ```fuzzing_stats.csv``` is created in the current directory.

## ~~How to run ```input-inspector.py```~~ (deprecated)
This script helps inspect the generated inputs. It prints them in various ways.

```
# To print error messages in each cluster
python input-inspector.py --inputdir=/path/to/workdir/expect_ok_prev_ok/XXX.yaml_workdir
```

```
# To print input-output pairs in each cluster
python input-inspector.py --inputdir=/path/to/workdir/expect_ok_prev_ok/XXX.yaml_workdir --printpairs
```

```
# To unpickle a XXX.p file
python input-inspector.py --inputdir=/path/to/workdir/expect_ok_prev_ok/XXX.yaml_workdir --pfile=XXX.p
```
