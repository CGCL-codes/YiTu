#####################################################
            GraphSAGE
#####################################################
###### ogbn-products  #####
# YiTu_GNN ogbn-products 
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main_without_pipeline.py --graph_name ogbn-products \
    --part_config /home/data/YiTu_GNN/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl ogbn-products
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-products \
    --part_config /home/data/dgl/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

###### ogbn-papers100M  #####
# YiTu_GNN ogbn-papers100M
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main_without_pipeline.py --graph_name ogbn-papers100M \
    --part_config /home/data/YiTu_GNN/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl ogbn-papers100M
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-papers100M \
    --part_config /home/data/dgl/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

###### soc-LiveJournal1  #####
# YiTu_GNN soc-LiveJournal1
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main_without_pipeline.py --graph_name soc-LiveJournal1 \
    --part_config /home/data/YiTu_GNN/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl soc-LiveJournal1
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name soc-LiveJournal1 \
    --part_config /home/data/dgl/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"


###### soc-pokec-relationships  #####
# YiTu_GNN soc-pokec-relationships
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main_without_pipeline.py --graph_name soc-pokec-relationships \
    --part_config /home/data/YiTu_GNN/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl soc-pokec-relationships
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name soc-pokec-relationships \
    --part_config /home/data/dgl/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"


#################################################
                GCN
#################################################

###### ogbn-products  #####
# YiTu_GNN ogbn-products 
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-products \
    --part_config /home/data/YiTu_GNN/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl ogbn-products
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-products \
    --part_config /home/data/dgl/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

###### ogbn-papers100M  #####
# YiTu_GNN ogbn-papers100M
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-papers100M \
    --part_config /home/data/YiTu_GNN/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl ogbn-papers100M
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-papers100M \
    --part_config /home/data/dgl/ogbn-papers100M/random/4part_data/ogbn-papers100M.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

###### soc-LiveJournal1  #####
# YiTu_GNN soc-LiveJournal1
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name soc-LiveJournal1 \
    --part_config /home/data/YiTu_GNN/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl soc-LiveJournal1
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name soc-LiveJournal1 \
    --part_config /home/data/dgl/soc-LiveJournal1/random/4part_data/soc-LiveJournal1.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"


###### soc-pokec-relationships  #####
# YiTu_GNN soc-pokec-relationships
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name soc-pokec-relationships \
    --part_config /home/data/YiTu_GNN/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"

# distdgl soc-pokec-relationships
python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/gcn \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name soc-pokec-relationships \
    --part_config /home/data/dgl/soc-pokec-relationships/random/4part_data/soc-pokec-relationships.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"