# out of core

> 单机处理大图

1. 实现的graphsage依赖`torch_scatter`, 首先安装：
```shell
docker exec -it YiTu_GNN bash
cd /home
git clone https://github.com/rusty1s/pytorch_scatter.git -b 2.0.6
cd pytorch_scatter
python setup.py install
```

2. 数据预处理：将图数据转换为YiTu_GNN支持的格式
```shell
docker exec -it YiTu_GNN bash
cd YiTu_GNN/examples/YiTu_GNN/out-of-core/
python preprocess.py --dataset ogbn-products --num_parts 2 --data_path /home/data --output_path /home/data/ogbn-products
```

3. 对图拓扑和图特征进行partition
```shell
python partition_graph.py --num_parts 2 --data_path /home/data/ogbn-products
```

4. 执行程序：
```shell

python main.py --data_path /home/data/output --num_workers 2
```