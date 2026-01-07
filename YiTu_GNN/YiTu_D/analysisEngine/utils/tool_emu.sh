#!/bin/bash
time_string=$(date "+%m%d-%H%M")
date_str=${time_string//:}
log_path=./ret_$3

TOTALPIPE=14
BASE=pax_hbmw_v2.2
APP=pr

for i in 8 9 10 11 12 13 #1 2 3 4 5 6 7 
do
    LITTLE=$i
	BIG=$[$TOTALPIPE-i]
	echo $i $[$TOTALPIPE-i]
	

	sed -i "/LITTLE_KERNEL_NUM/c LITTLE_KERNEL_NUM=$LITTLE" ./global_para.mk
	sed -i "/BIG_KERNEL_NUM/c BIG_KERNEL_NUM=$BIG" ./global_para.mk

	make exe
	
	./utils/tool_test.sh ../${BASE}_${LITTLE}v${BIG}/xclbin_hw_$APP/graph_fpga.hw.xilinx_u280_xdma_201920_3.xclbin pr ${LITTLE}v${BIG}

done


