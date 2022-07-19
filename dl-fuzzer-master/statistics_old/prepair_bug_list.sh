#!/bin/bash

home='/home'
expect_ok="expect_ok"
expect_excpt="expect_exception"
prev_ok="prev_ok"
permute="permute"

workdir=$1
obey=$2
compute_permute=$3
constr=$4
adapt=$5
if [ $# -ne 5 ]
  then
    echo "Need 5 arguments:
          1) path to workdir;
          2) [expect_ok | expect_exception];
          3) flag to compute permute or not [true | false];
          4) [constr | no_constr];
          5) [adapt | no_adapt]"
    exit 1
fi
if [ "$obey" == "$expect_ok" ]; then
    subdir=$workdir/$expect_ok
else
    subdir=$workdir/$expect_excpt
fi
#prev_ok_dir=$subdir"_prev_ok"
prev_ok_dir=$subdir"_"$constr"_"$adapt
permute_dir=$subdir"_permute"

pushd $prev_ok_dir > /dev/null
tmp_api=$workdir/tmp_api
tmp_type=$workdir/tmp_type
tmp_input=$workdir/tmp_input
echo "Extracting API names"
find . -name '*_script_record' -not -iname 'failure*' | sort -k 1 | cut -d"/" -f2 | sed 's/.yaml_workdir//g' > $tmp_api  #extract API name
echo "Extracting bug types"
find . -name '*_script_record' -not -iname 'failure*' | sort -k 1 | cut -d"/" -f3 | sed 's/_script_record//g' > $tmp_type #extract bug tpye
echo "Extracting input count and input script path"
find . -name '*_script_record' -not -iname 'failure*' -type f -print0 | wc -l --files0-from=- | sort -k 2 | sed '$ d' | sed "s~ \.~ $prev_ok_dir~g" > $tmp_input #extract input count and input script
popd > /dev/null
tmp_1=$workdir/tmp_1
tmp_2=$workdir/tmp_2
# append columns
echo "Creating prev_ok bug list"
bug_list_prev=$workdir/bug_list_${obey}_prev
paste -d " " $tmp_api <(awk '{print $0}' $tmp_type) > $tmp_1 
paste -d " " $tmp_1 <(awk '{print $0}' $tmp_input) > $tmp_2
echo "API Type Count Input" | cat - $tmp_2 > $bug_list_prev #add header
rm -f $tmp_api $tmp_type $tmp_input $tmp_1 $tmp_2

if [ "$compute_permute" == "true" ]; then
  echo "Computing diff between prev_ok and permute"
  pushd $prev_ok_dir > /dev/null
  find . -name '*_script_record' -not -iname 'failure*' | sort -k 1 > $tmp_1
  popd > /dev/null

  pushd $permute_dir > /dev/null
  find . -name '*_script_record' -not -iname 'failure*' | sort -k 1 > $tmp_2
  
  prev_only=`diff $tmp_1 $tmp_2 | grep '<' |  wc -l`
  permute_only=`diff $tmp_1 $tmp_2 | grep '>' |  wc -l`
  both=`comm -1 -2 $tmp_1 $tmp_2 | wc -l`
  echo "prev_ok only: $prev_only, permute only: $permute_only, both: $both" >> $bug_list_prev
  echo "prev_ok only: $prev_only, permute only: $permute_only, both: $both"

  if [ $(($permute_only)) -gt 0 ]; then
    echo "Creating permute only bug list"
    bug_list_per=$workdir/bug_list_${obey}_permute_only
    diff $tmp_1 $tmp_2 | grep '>' | sed "s/> //g" > $bug_list_per
    cat $bug_list_per | cut -d"/" -f2 | sed 's/.yaml_workdir//g' > $tmp_api 
    cat $bug_list_per | cut -d"/" -f3 | sed 's/_script_record//g' > $tmp_type 
    awk '{print $0}' $bug_list_per | xargs wc -l | sort -k 2 | sed '$ d' | sed "s~ \.~ $permute_dir~g" | tr -s " " > $tmp_input

    paste -d " " $tmp_api <(awk '{print $0}' $tmp_type) > $tmp_1 
    paste -d " " $tmp_1 <(awk '{print $0}' $tmp_input) > $bug_list_per
  fi
  popd > /dev/null
  rm -f $tmp_api $tmp_type $tmp_input $tmp_1 $tmp_2
fi
