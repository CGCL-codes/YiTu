#!/bin/bash
time_string=$(date "+%m%d-%H%M")
date_str=${time_string//:}
#log_path=./ret_$3



if [ $# -lt 2 ];
  then
    echo "[FAILD] missing arguments"
    echo "eg. ------>"
    echo "./compile.sh #BIG_KERNEL #LITTLE_KERNEL "
    exit -1
fi

# $1 xclbin
# $2 app
# $3 notes

echo "$1 $2"

cp -r ../ReGraph  ../ReGraph$1v$2

cd ../ReGraph$1v$2

make cleanall

sed -i "/LITTLE_KERNEL_NUM/c LITTLE_KERNEL_NUM=$1" ./global_para.mk
sed -i "/BIG_KERNEL_NUM/c BIG_KERNEL_NUM=$2" ./global_para.mk

make autogen

nohup make all &



