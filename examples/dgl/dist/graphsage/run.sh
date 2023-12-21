python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/dgl/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/dgl/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-products \
    --part_config /home/data/dgl/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"
