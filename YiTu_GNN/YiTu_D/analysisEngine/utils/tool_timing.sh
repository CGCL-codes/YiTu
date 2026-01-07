log_file=""
if [ -e "_x/link/vivado/vivado.log" ]
then
log_file="_x/link/vivado/vivado.log"
fi

if [ -e "_x/link/vivado/vpl/vivado.log" ]
then
log_file="_x/link/vivado/vpl/vivado.log"
fi



if [ -z ${log_file} ]; then
    echo "no log file"
else
	cat ${log_file}  | grep TNS
	cat ${log_file}  | grep "The frequency is being automatically changed to"
	cat ${log_file}  | grep scaled
	
fi
