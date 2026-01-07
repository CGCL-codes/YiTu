#!/bin/bash

prj_file=""
if [ -e "_x/link/vivado/prj/prj.xpr" ]
then
prj_file=" _x/link/vivado/prj/prj.xpr"
fi

if [ -e "_x/link/vivado/vpl/prj/prj.xpr" ]
then
prj_file=" _x/link/vivado/vpl/prj/prj.xpr"
fi



if [ -z ${prj_file} ]; then
    echo "no prj file"
else
	vivado -mode batch -source utils/report_usage.tcl -tclargs ${prj_file}
fi
