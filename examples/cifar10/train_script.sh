#!/bin/bash
#set -e
#set -x

folder="examples/cifar10"


if [ "$#" -lt 5 ]; then
	echo "Illegal number of parameters"
	echo "Usage: train_script network_prefix device_id template_file exp_id file_suffix"
	exit
fi

network_prefix=$1
device_id=$2
template_file=$3
exp_id=$4
file_suffix=$5

snapshot_path=$folder/snapshot

solverfile=$snapshot_path/${network_prefix}_${exp_id}_${file_suffix}_solver.prototxt

cat $folder/${template_file} > $solverfile
echo "net: \"${folder}/${network_prefix}_train_test.prototxt\"" >> $solverfile
echo "snapshot_prefix: \"$snapshot_path/${network_prefix}_${file_suffix}_${exp_id}\"" >> $solverfile
echo "device_id: ${device_id} " >> $solverfile
./build/tools/caffe train --solver=$solverfile  2>&1 | tee  outputFile/${network_prefix}_${file_suffix}_${exp_id}.out

