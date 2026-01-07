date_str=`date +%Y%m%d%T`


release_path=release_${date_str}
mkdir -p ${release_path}

cp -r ./_x/link/int      		./${release_path}
cp -r ./_x/reports       		./${release_path}
cp _x/link/vivado/vivado.log  	./${release_path}
cp host_graph_fpga              ./${release_path}


git status > 				${release_path}/git_status.log
git diff > 					${release_path}/code_diff.diff
git diff --cached > 		${release_path}/code_cached.diff
git log --graph  -10 > 		${release_path}/git_log.log
git show HEAD > 			${release_path}/git_show.diff




