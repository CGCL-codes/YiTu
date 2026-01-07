#!/bin/sh
echo > ./timing_report.txt

for dir in `ls ../`; do                  
    if [ -d $dir ] 
    then
	cd $dir;
    	echo $dir >> ../timing_report.txt
    	#make clean;
		./utils/tool_timing.sh | grep "MHz" >> ../timing_report.txt
    	cd ..;
    fi
done
