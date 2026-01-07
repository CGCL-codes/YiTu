#!/bin/bash

tmp_string=`date +%Y%m%d%T`
date_str=${tmp_string//:}
log_path=test_log_${date_str}_$2

mkdir -p ${log_path}


if [ $# -lt 2 ];
  then
    echo "[FAILD] missing config for start test"
    echo "eg. ------>"
    echo "./tool_test.sh xx.xclbin cc  "
    exit -1
fi


# $1 xclbin
# $2 app

DATASET=(   'rmat-19-32.txt' \
            'graph500-scale25-ef16_adj.edges'\
)

make app=$2 exe 

for dataset  in "${DATASET[@]}"
do
echo "/graph_data/$dataset"

./host_graph_fpga_$2 ~/Dropbox/Experiment/$1/$2_$1.xclbin /graph_data/$dataset > ./${log_path}/$2_$dataset.log

done


