#!/bin/bash

# $1 xclbin
# $2 app
DATASET=(   'rmat-19-32.txt' \
            'rmat-21-32.txt' \
            'rmat-24-16.txt' \
            'bio-mouse-gene.edges' \
            'web-Google.mtx'\
            'wiki-Talk.txt'\
            'amazon-2008.mtx' \
            'web-hudong.edges' \
            'web-baidu-baike.edges' \
            'wiki-topcats.mtx' \
            'soc-flickr-und.edges' \
            'pokec-relationships.txt' \
            'LiveJournal1.txt' \
            'soc-twitter-2010.mtx' \
            'wikipedia-20070206.mtx' \
            'ca-hollywood-2009.mtx' \
            'graph500-scale23-ef16_adj.edges' \
            'graph500-scale24-ef16_adj.edges' \
            'graph500-scale25-ef16_adj.edges' \
)

for dataset  in "${DATASET[@]}"
do
echo "$dataset"
cat $1/$2_$dataset.log | grep  $3

done