#!/bin/bash
#usage
#bash remove_fail_workdir.sh /path/to/docker/workdir [expect_ok|expect_exception] [prev_ok|permute]
work_home=$1
obey=$2 #[expect_ok | expect_exception]
if [ $# -ne 2 ]
  then
    echo "Need 2 arguments: 1) path to workdir, 2) expect_ok or expect_exception"
    exit 0
fi
expect_ok="expect_ok"
expect_excpt="expect_exception"
if [ "$obey" == "$expect_ok" ]; then
  subdir=$expect_ok"_prev_ok"
else
  subdir=$expect_excpt"_prev_ok"
fi
workdir=$work_home"/"$subdir
#pushd $work_home > /dev/null
while read -r script_record 
do
  echo "Working on $script_record"
  out=$script_record".out"
  bash $script_record > $out 2>&1 
done < <(find $workdir -name 'Abort_script_record' | sort -n)
#popd > /dev/null
