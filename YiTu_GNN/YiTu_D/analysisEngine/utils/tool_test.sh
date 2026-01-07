#!/bin/bash
time_string=$(date "+%m%d-%H%M")
date_str=${time_string//:}
log_path=./ret_$3

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

DATASET=(   
            'rmat-19-32.txt' \
            'bio-mouse-gene.edges'\
            'web-Google.mtx'\
            'orkut.ungraph.txt'\
            'amazon-2008.mtx' \
            'web-hudong.edges' \
            'web-baidu-baike.edges' \
            'wiki-topcats.mtx' \
            'soc-flickr.ungraph.edges' \
            'pokec-relationships.txt' \
            'rmat-21-32.txt' \
            'rmat-24-16.txt' \
            'LiveJournal.ungraph.txt'
            'LiveJournal1.txt' \
            'wikipedia-20070206.mtx' \
            'ca-hollywood-2009.ungraph.mtx' \
            'graph500-scale23-ef16_adj.edges' \
            'dbpedia-link.txt' \
            # 'soc-twitter-2010.mtx' \
)

echo > ./${log_path}/SUMMARY.log
echo > ./${log_path}/PROFILE.log

for dataset  in "${DATASET[@]}"
do
echo "$dataset"

./host_graph_fpga_$2 $1 $dataset > ./${log_path}/$dataset.log

cat ./${log_path}/$dataset.log | grep 'e2e\|error'
cat ./${log_path}/$dataset.log | grep 'e2e\|error' >> ./${log_path}/SUMMARY.log

echo "$dataset" >> ./${log_path}/PROFILE.log
cat ./${log_path}/$dataset.log | grep 'PROFILE' >> ./${log_path}/PROFILE.log

done


