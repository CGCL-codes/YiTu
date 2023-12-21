# workspace: 设置本地的工作目录
# -w：设置docker中的工作目录
# part_config: 图partition的配置文件所在位置（docker路径）
# ip_config: ip_config所在路径（本地路径）
/home/nx/anaconda3/envs/distdgl/bin/python YiTu_GNN/utils/launch.py   \
		--workspace /home/nx/ningxin/DistGNN/YiTu_GNN  \
		--num_trainers 1   \
		--num_samplers 0   \
		--num_servers 1   \
		--part_config /home/data/YiTu_GNN/ogb-products/random/4part_data/ogbn-products.json   \
		--ip_config examples/ip_config.txt  \
		"docker exec --env PYTHONPATH=/home/YiTu_GNN --env NCCL_IB_DISABLE=1 --env NCCL_DEBUG=INFO -w /home/YiTu_GNN YiTu_GNN /opt/conda/bin/python examples/YiTu_GNN/graphsage/main.py \
        --graph_name ogbn-products --ip_config examples/ip_config.txt --num_epochs 2"

# echo $1
# echo $2
# ## 不使用docker启动服务端和客户端
# # 服务端
# # ssh -o StrictHostKeyChecking=no -p 22 $1 \
# # "cd /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN; \
# # (export DGL_ROLE=server DGL_NUM_SAMPLER=0 OMP_NUM_THREADS=1 \
# # DGL_NUM_CLIENT=2 DGL_CONF_PATH=/home/nx/ningxin/.data/p3/2part_data/ogbn-products.json \
# # DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc  DGL_SERVER_ID=0; \
# # /home/nx/anaconda3/envs/distdgl/bin/python  /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN/main.py \
# # --graph_name ogbn-products --ip_config ip_config.txt --part_config /home/nx/ningxin/.data/p3/2part_data/ogbn-products.json)" &


# # cd /home/YiTu_GNN; \
# # (export DGL_ROLE=server DGL_NUM_SAMPLER=0 OMP_NUM_THREADS=1 \
# # DGL_NUM_CLIENT=2 DGL_CONF_PATH=/home/2part_data/ogbn-products.json \
# # DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc  DGL_SERVER_ID=1 \
# # NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL; \
# # python main.py --graph_name ogbn-products --ip_config ip_config.txt --part_config /home/2part_data/ogbn-products.json)

# ## 客户端
# # ssh -o StrictHostKeyChecking=no -p 22 $1 \
# # "cd /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN; \
# # (export DGL_DIST_MODE=distributed DGL_ROLE=client DGL_NUM_SAMPLER=0 \
# # DGL_NUM_CLIENT=2 DGL_CONF_PATH=/home/nx/ningxin/.data/p3/2part_data/ogbn-products.json \
# # DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc OMP_NUM_THREADS=28 ; \
# # /home/nx/anaconda3/envs/distdgl/bin/python -m torch.distributed.launch \
# # --nproc_per_node=1 --nnodes=2 --node_rank=$2 --master_addr=11.11.11.18 --master_port=1234  \
# # /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN/main.py --graph_name ogbn-products --ip_config ip_config.txt  \
# # --part_config /home/nx/ningxin/.data/p3/2part_data/ogbn-products.json)" &

# ## 服务端和客户端位于同一个docker
# # 运行容器
# # ssh -o StrictHostKeyChecking=no -p 22 $1 \
# # "sudo docker run --gpus all -v /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN/:/home/YiTu_GNN -v /home/nx/ningxin/.data/p3/2part_data/:/home/2part_data \
# # --rm -dit --shm-size="5g" --network=host --name YiTu_GNN dgl /bin/bash" \

# # 服务端
# ssh -o StrictHostKeyChecking=no -p 22 $1 \
# "docker exec \
# --env DGL_ROLE=server --env DGL_NUM_SAMPLER=0 --env OMP_NUM_THREADS=1 --env DGL_NUM_CLIENT=2 \
# --env DGL_CONF_PATH=/home/2part_data/ogbn-products.json --env DGL_IP_CONFIG=/home/YiTu_GNN/ip_config.txt \
# --env DGL_NUM_SERVER=1 --env DGL_GRAPH_FORMAT=csc  --env DGL_SERVER_ID=$2 \
# YiTu_GNN /opt/conda/bin/python /home/YiTu_GNN/main.py --graph_name ogbn-products --ip_config /home/YiTu_GNN/ip_config.txt --part_config /home/2part_data/ogbn-products.json" &

# # 客户端
# ssh -o StrictHostKeyChecking=no -p 22 $1 \
# "docker exec \
# --env DGL_DIST_MODE=distributed --env DGL_ROLE=client --env DGL_NUM_SAMPLER=0 --env DGL_NUM_CLIENT=2 \
# --env DGL_CONF_PATH=/home/2part_data/ogbn-products.json --env DGL_IP_CONFIG=/home/YiTu_GNN/ip_config.txt \
# --env DGL_NUM_SERVER=1 --env DGL_GRAPH_FORMAT=csc --env OMP_NUM_THREADS=28 \
# YiTu_GNN /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=$2 --master_addr=11.11.11.18 --master_port=1234 \
# /home/YiTu_GNN/main.py --graph_name ogbn-products --ip_config /home/YiTu_GNN/ip_config.txt  --part_config /home/2part_data/ogbn-products.json" &

# # 服务端和客户端位于不同容器
# # 服务端
# # ssh -o StrictHostKeyChecking=no -p 22 $1 \
# # "docker run --gpus all -v /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN/:/home/YiTu_GNN -v /home/nx/ningxin/.data/p3/2part_data/:/home/2part_data --rm -dit --network=host \
# # --env DGL_ROLE=server --env DGL_NUM_SAMPLER=0 --env OMP_NUM_THREADS=1 --env DGL_NUM_CLIENT=2 \
# # --env DGL_CONF_PATH=/home/2part_data/ogbn-products.json --env DGL_IP_CONFIG=/home/YiTu_GNN/ip_config.txt \
# # --env DGL_NUM_SERVER=1 --env DGL_GRAPH_FORMAT=csc  --env DGL_SERVER_ID=$2 \
# # dgl /opt/conda/bin/python /home/YiTu_GNN/main.py --graph_name ogbn-products --ip_config /home/YiTu_GNN/ip_config.txt --part_config /home/2part_data/ogbn-products.json" &

# # 客户端:训练器
# # ssh -o StrictHostKeyChecking=no -p 22 $1 \
# # "docker run --gpus all -v /home/nx/ningxin/nfs/YiTu_GNN/YiTu_GNN/:/home/YiTu_GNN -v /home/nx/ningxin/.data/p3/2part_data/:/home/2part_data --rm --network=host \
# # --env DGL_DIST_MODE=distributed --env DGL_ROLE=client --env DGL_NUM_SAMPLER=0 --env DGL_NUM_CLIENT=2 \
# # --env DGL_CONF_PATH=/home/2part_data/ogbn-products.json --env DGL_IP_CONFIG=/home/YiTu_GNN/ip_config.txt \
# # --env DGL_NUM_SERVER=1 --env DGL_GRAPH_FORMAT=csc --env OMP_NUM_THREADS=28 \
# # dgl /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=$2 --master_addr=11.11.11.18 --master_port=1234 \
# # /home/YiTu_GNN/main.py --graph_name ogbn-products --ip_config /home/YiTu_GNN/ip_config.txt  --part_config /home/2part_data/ogbn-products.json" &

# wait