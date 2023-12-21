/home/nx/anaconda3/envs/distdgl/bin/python /home/nx/ningxin/DistGNN/YiTu_GNN/YiTu_GNN/utils/launch.py   \
		--workspace /home/nx/ningxin/DistGNN/YiTu_GNN  \
		--num_trainers 1   \
		--num_samplers 0   \
		--num_servers 1   \
		--part_config /home/data/YiTu_GNN/ogbn-products/random/2part_data/ogbn-products.json   \
		--ip_config examples/ip_config.txt  \
		"docker exec --env PYTHONPATH=/home/YiTu_GNN --env NCCL_IB_DISABLE=1 --env NCCL_DEBUG=INFO -w /home/YiTu_GNN YiTu_GNN /opt/conda/bin/python examples/YiTu_GNN/dist/graphsage/main_without_pipeline.py \
        --graph_name ogbn-products --ip_config examples/ip_config.txt \
		--part_config /home/data/YiTu_GNN/ogbn-products/random/2part_data/ogbn-products.json --num_epochs 5 --eval_every 2 --num_hidden 16"
# --env NCCL_DEBUG_SUBSYS=ALL