#!/bin/bash
#### usage
# $ bash run_batch.sh /absolute path/to/constraints/constraints_1/changed [expect_ok|expect_exception] [prev_ok|permute] [tensorflow|pytorch|mxnet]

home='/home'
expect_ok="expect_ok"
expect_excpt="expect_exception"
prev_ok="prev_ok"
permute="permute"

constraints_folder=$1
obey=$2
adapt=$3
package=$4
constr=$5
adaptive_gen=$6
if [ $# -ne 6 ]
  then
    echo "Need 6 arguments: 1) path to constraints folder, 2) obey option [expect_ok|expect_exception], 3) adapt_to option [prev_ok|permute], 4) python package [tensorflow|pytorch|mxnet], 5) following constraints [ constr | no_constr], and 6) enabling adaptive generation [adapt | no_adapt]." 
    exit 0
fi
if [[ "$package" != "tensorflow" && "$package" != "pytorch" && "$package" != "mxnet" ]];then
  echo "Error: unsupported package choice: $package"
  exit 1
fi
if [ "$obey" == "$expect_ok" ]; then
    subdir=$expect_ok"_"$constr"_"$adaptive_gen
else
    subdir=$expect_excpt"_"$constr"_"$adaptive_gen
fi
WORKDIR="$home"/workdir/"$subdir"
workorder_record="$WORKDIR"/"record_work"
division_id=$( basename "$constraints_folder" )
division_record="$WORKDIR"/"${division_id}_record"
if [ -f "$workorder_record" ]; then
  echo "... record_work exists ..."
else
  mkdir -p "$WORKDIR"
  touch "$workorder_record"
  touch "$division_record"
fi

FUZZERGIT="$home"/code/dl-fuzzer

for c in "$constraints_folder"/*; do
  name=$( basename "$c" )
  if grep -qw "$name" "$workorder_record"; then
    echo "Skipping $name"
    continue
  else
    echo "$name" >> $workorder_record
    echo "$name" >> "$division_record"
  fi

  workdir="$WORKDIR/${name}_workdir"
  mkdir -p "$workdir"
  echo "$workdir"
  dump_folder="$workdir"/".dump"
  mkdir -p "$dump_folder"
  args=(
    "$c"
    "${FUZZERGIT}/${package}/${package}_dtypes.yml"
    "--max_iter=1000"
    "--workdir=$workdir"
    "--cluster"
    "--dist_threshold=0.5"
    "--dist_metric=jaccard"
    "--adapt_to=$adapt"
    "--fuzz_optional"
    "--gen_script"
    "--timeout=10"
  )
  if [ "$package" == "pytorch" ];then
    args+=("--data_construct=tensor")
  elif [ "$package" == "mxnet" ];then
    args+=("--data_construct=nd.array")
  fi
  if [ "$obey" == "$expect_ok" ]; then
    args+=("--obey")
  fi
  if [ "$constr" == "no_constr" ]; then
    args+=("--ignore")
  fi
  if [ "$adaptive_gen" == "adapt" ];then
    args+=("--consec_fail=10")
  else
    args+=("--consec_fail=2000")
  fi
  pushd "$dump_folder" > /dev/null
  echo "running in"
  pwd
  python ${FUZZERGIT}/fuzzer/fuzzer-driver.py "${args[@]}" &> $workdir/out
  popd > /dev/null
  rm -r "$dump_folder"
  status=$?
  if [ $status -eq 0 ];then
    status_msg="ok"
  else
    status_msg="fail"
  fi
  echo "===  $( basename "$c" ) ===    $status_msg"
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
