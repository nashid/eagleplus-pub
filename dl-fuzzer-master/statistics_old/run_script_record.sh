#!/bin/bash
#usuage: bash run_script_record.sh /path/to/timeout_script_record
to_run=$1 #collectino of timeout inputs
if [ $# -ne 1 ]
  then
    echo "Need 1 argument: 1) path to file that collects python inputs to run" 
    exit 0
fi

while read -r line 
do
  echo -e "\n\n######################################################"
  echo "R U N N I N G $line"
  timeout --kill-after=100 --signal=KILL 100 $line
  status=$?
  echo "Status Code: $status"
done < "$to_run"

