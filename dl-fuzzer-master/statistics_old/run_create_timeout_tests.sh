#!/bin/bash
#usage example
#  bash run_create_timeout_tests.sh /local1/m346kim/dl-fuzzing/docker_4_f19509/timeouts_prev_ok api_list
#  bash run_create_timeout_tests.sh /local1/m346kim/dl-fuzzing/docker_4_f19509/timeouts_prev_ok api_list all all
#  bash run_create_timeout_tests.sh /local1/m346kim/dl-fuzzing/docker_4_f19509/timeouts_prev_ok api_list all one /local1/someoneelse 
#  bash run_create_timeout_tests.sh /local1/m346kim/dl-fuzzing/docker_4_f19509/timeouts_prev_ok api_list all all /local1/someoneelse  

out_home=$1 #home directory path to where timeout python inputs are stored
api_list=${2:all} #file name to where the list of APIs to run are stored. This file should be located under $out_home. If provided as "all", then it will run all APIs
run_all_flag=${3:one} #expect "all". If not provided, default is create a test from only one timeout warning.
write_home=$4 # if not previded, it's same as out_home


grep_cmd="grep -e ."
if [ "$api_list" != "all" ];
then
  while IFS= read -r line
  do
    grep_cmd=$grep_cmd" -e $line"
  done < "$api_list"  
fi

if [ "$run_all_flag" == "all" ];
then
  find_cmd="find ${out_home} -name *.timeout*.py"
else
  find_cmd="find ${out_home} -name *.timeout1.py" #pick only one test for each api
fi

if [ -z "$write_home" ];
then
  write_home=$out_home
fi

file_lists=()
while IFS=  read -r -d $'\0'; do
    file_lists+=("$REPLY")
done < <(${find_cmd} | ${grep_cmd} | tr "\n" "\0" | sort -z)

workorder_record="$write_home"/"record_work_all"
if [ -f "$workorder_record" ]; then
  echo "... record_work exists ..."
else
  touch "$workorder_record"
fi

for test in ${file_lists[@]}
do
  if grep -qw "$test" "$workorder_record"; then
    echo "Skipping $test"
    continue
  else
    echo "$test" >> $workorder_record
  fi

  dirpath=$(dirname "${test}") 
  log=${dirpath}"/out" ##/local1/m346kim/dl-fuzzing/docker_4_f19509/timeouts_prev_ok/tf.broadcast_to/log

  echo "============== Executing ${test}  ================"
  echo "============== Executing $(basename "${test}")  ================" >> $log
  python3 ${test} >> $log 2>&1
 <<'user_input'
  echo "continue? [y/n]"
  read -r go_on
  if [ "$go_on" == "y" ]; then
    continue
  else  # anything other than 'y' will break
    break
  fi
user_input
done

