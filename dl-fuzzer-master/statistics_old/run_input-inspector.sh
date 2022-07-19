#!/bin/bash

#usage bash run_input-inspector.sh /path/to/workdir /path/to/failure_list_file
workdir=$1
target_list_file=$2

# read the target file
unset lines
while IFS= read -r line;
do
  yaml_dir=$workdir"/"$line".yaml_workdir"
  echo "============================ =========" 
  echo $yaml_dir
  python input-inspector.py --inputdir=$yaml_dir

  status=$?
  if [[ $status -ne 0 ]];then
    echo "========= No input dir exists =========" 
  fi
done < "$target_list_file"
