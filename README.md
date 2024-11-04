# YiTu
YiTu is an easy-to-use runtime designed to fully exploit hybrid parallelism across various hardware platforms (e.g., GPUs) to efficiently execute a wide range of graph algorithms, including graph neural networks (GNNs). It offers optimized support for heterogeneous and temporal graphs, enabling advanced analysis on complex, diverse, and dynamic graph structures. YiTu includes four modules: YITU_H, which handles heterogeneous graphs; YITU_T, focused on temporal graph analysis; YITU_GNN, dedicated to the implementation of standard Graph Neural Networks (GNNs); and YITU_GP, which offers advanced tools for graph analysis. Each module is designed to address specific aspects of graph processing and analysis, enabling efficient handling of diverse graph types and tasks.Next are the specific descriptions of each module.

<div align=center>
<img src="https://github.com/czk23/YiTu/blob/main/images/system%20design.png">
</div>

## Introduction

### YITU_GNN
YITU_GNN is a specialized module within YiTu that focuses on the implementation and analysis of standard Graph Neural Networks (GNNs). It allows users to efficiently build and experiment with popular GNN architectures, such as Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT). With a flexible API, YITU_GNN enables easy customization of models and optimizes performance for large-scale graph training. By integrating with other modules in YiTu, it enhances the system's overall capabilities, making it a powerful tool for tasks like node classification, link prediction, and graph classification, ultimately facilitating deeper insights into graph-structured data.

### YITU_GP
YITU_GP is a dedicated module within YiTu that specializes in graph analysis functionalities. This module provides users with advanced tools and algorithms for extracting insights from graph-structured data, enabling a wide range of analytical tasks. With YITU_GP, users can perform various operations, including community detection, centrality measures, and so on. The module is designed to handle large graphs efficiently, ensuring that users can analyze complex relationships and patterns within their data. By integrating seamlessly with other modules in YiTu, YITU_GP enhances the overall analytical capabilities of the system, making it an essential component for applications requiring in-depth graph analysis and interpretation.

### YITU_T
YITU_T is a specialized module within YiTu that focuses on the functionalities related to temporal graphs. It enables the processing and analysis of time-evolving graph data, allowing for the exploration of dynamic relationships and changes over time. With YITU_T, users can efficiently manage temporal aspects of graphs, supporting tasks such as event forecasting, trend analysis, and social network dynamics. This module ensures that users can effectively handle the complexities of temporal data, providing tools to analyze how graph structures evolve and interact in real-time. By integrating these capabilities, YITU_T enhances YiTu’s overall functionality, making it a powerful tool for applications requiring advanced temporal graph analysis.

### YITU_H
YITU_H is a dedicated module within YiTu that focuses on the functionalities related to heterogeneous graphs. It enables the handling of graphs with multiple types of nodes and edges, allowing for a rich representation of complex relationships. With YITU_H, users can efficiently manage diverse graph structures, making it particularly suitable for applications such as recommendation systems, knowledge graphs, and multi-relational data analysis. This module ensures seamless integration and processing of various graph elements, providing tools to explore and analyze the interactions among different types of entities. By enhancing YiTu’s capabilities in dealing with heterogeneous graphs, YITU_H empowers users to unlock deeper insights and drive advanced analyses across diverse applications.

## Installation

Provide two installation methods:
1. Build docker image
2. Install with conda

### Building Docker Image

1. First, build a virtual environment with Docker, and YiTu_GNN can run on the environment. Docker installation please [reference](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide). 

By running the following commands, we can construct a docker image called `YiTu_GNN`, on which all subsequent experiments will be performed.
```shell
cd YiTu_GNN/docker
docker build -f Dockerfile -t YiTu_GNN .
```
2. starting docker container:
```shell
docker run --gpus all -e NCCL_SOCKET_IFNAME=eno1 --rm -dit --shm-size="5g" --network=host --name YiTu_GNN YiTu_GNN /bin/bash
```
* --gpus all：indicating that all GPUs will be used
* -e NCCL_SOCKET_IFNAME=eno1: setting the Network Interface Card to be used
* --shm-size="5g": setting the size of shared memory
* --network=host: indicating that host network will be used

3. installing YiTu_GNN in docker container:
```shell
# 1. copy YiTu_GNN to docker container
docker cp YiTu_GNN YiTu_GNN:/home
# 2. enter docker container
docker exec -it YiTu_GNN bash
cd /home/YiTu_GNN
# 3. install YiTu_GNN
python setup.py install
```

### Installing with Conda

1. installing cmake:
```shell
version=3.18
build=0
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build-Linux-x86_64.sh 
sudo mkdir /opt/cmake
sudo sh cmake-$version.$build-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
cd ~
rm -rf ~/temp
```
2. installing conda:
```shell
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export PATH=/opt/conda/bin:$PATH

apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

export TINI_VERSION=v0.16.1
source ~/.bashrc
```
3. installing related dependency packages：
```shell
conda create -n YiTu_GNN python=3.7.5
conda activate YiTu_GNN
conda install -y astunparse numpy ninja pyyaml mkl \
	mkl-include setuptools cffi \
	typing_extensions future six \
	requests dataclasses \
	pytest nose cython scipy \
	networkx matplotlib nltk \
	tqdm pandas scikit-learn && \
	conda install -y -c pytorch magma-cuda102
conda install pytest -c conda-forge
pip3 install rdflib
# stand-alone dependency packages
apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libgoogle-perftools-dev \
    protobuf-compiler && \
	rm -rf /var/lib/apt/lists/*
```
4. installing pytorch from source
```shell
mkdir ~/temp
cd ~/temp
# download the PyTorch source code that can support parameters with multiple versions
git clone --recursive https://github.com/Ningsir/pytorch.git -b multiversion-
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# Compile and install PyTorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
USE_NINJA=OFF python setup.py install --cmake

cd ~
rm -rf ~/temp
```

> Possible problem: Lack of valgrind.h
>
> Solution: cd third_party/valgrind && git checkout VALGRIND_3_18_0

5. installing DGL:
```shell
conda install -y -c dglteam dgl-cuda10.2=0.7.1
```
6. installing YiTu_GNN
```shell
python setup.py install
```
7. To use the heterogeneous graph training module, run the following commands on ./YiTu_GNN/YiTu_H directory:
```shell
python setup.py install
```
8. To use the Temporal GNN training module, run the following commands on ./YiTu_GNN/YiTu_T directory:
```shell
python setup.py build_ext --inplace
```

## Compilation

Environment requrements:
1. G++==7.5.0
2. cuda==10.2
3. Openmpi==2.1.1
4. Python=3.7

Run the following commands on root/examples/YiTu_GNN/NDP directory:

```shell
#compile graph algorithms with NDP
cmake .
make
#compile graph algorithms without NDP
cd nondp
make
cd ..
#compile graph formats converters
cd tools
make
cd ..
```

## Data Preprocessing

### 1. YiTu_GNN

#### Without NDP

##### Downloading Dataset:

```shell
cd /home/data
git clone https://github.com/kimiyoung/planetoid.git
```

##### Preprocessing:

```shell
python example/YiTu_GNN/single/data_process.py --data_path /home/data/planetoid/data --dataset citeseer
python example/YiTu_GNN/single/data_process.py --data_path /home/data/planetoid/data --dataset cora
python example/YiTu_GNN/single/data_process.py --data_path /home/data/planetoid/data --dataset pubmed
```

#### With NDP

##### Dataset Format (`vnum` represents the number of vertices)
  
* `adj.npz`: graph adjacency matrix, shape `(vnum, vnum)`, saved as `scipy.sparse` coo matrix format.
* `labels.npy`: vertex label, shape `(vnum,)`, saved as `numpy.array` format.
* `test.npy`, `train.npy`, `val.npy`: boolean array, shape `(vnum,)`, saved as `numpy.array` format.
* `feat.npy`(optional): feature of vertex, shape `(vnum, feat-size)`, saved as `numpy.array`, can be generated randomly if not provided.

##### Downloading Dataset and Converting Dataset Format
  
  ```shell
  mkdir reddit
  curl --output reddit/reddit.zip https://data.dgl.ai/dataset/reddit.zip
  unzip -d reddit reddit/reddit.zip
  python examples/YiTu_GNN/NDP/data/dgl2NDP.py --dataset reddit --out-dir /home/reddit
  ```

##### Subgraph generation
  
* hash partition
  
  ```shell
  python YiTu_GNN/NDP/partition/hash.py --num-hops 2 --partition 1 --dataset /home/reddit
  ```
* dg-based partition
  
  ```shell
  python YiTu_GNN/NDP/partition/dg.py --num-hops 2 --partition 1 --dataset /home/reddit
  ```

### 2. YiTu_GP

#### Input Graph Formats

The followings are two graph file examples.

Graph.el ("SOURCE DESTINATION" for each edge in each line):
```
0 1
0 3
2 3
1 2
```

Graph.wel ("SOURCE DESTINATION WEIGHT" for each edge in each line):
```
0 1 26
0 3 33
2 3 40
1 2 10
```

#### Graph Formats Converters

YiTu_GP accepts binary serialized pre-built CSR graph representation (.bcsr and .bcsrw). Reading binary formats is faster and more space efficient.
To convert edge-list (.el) and weighted edge-list (.wel) format graph files to the binary format, run the following commands on the root/examples/YiTu_GNN/NDP directory:
```
#convert Graph.el to Graph.bcsr
tools/converter path_to_Graph.el

#convert Graph.wel to Graph.bcsr and Graph.bcsrw
tools/converter path_to_Graph.wel
```
The first command converts Graph.el to the binary CSR format and generates a binary graph file with .bcsr extension under the same directory as the original file. The second command converts Graph.wel to a binary graph file with .bcsr extension and a binary edgeWeight file with .bcsrw extension.

### 3. YiTu_H

YiTu_H supports a diverse range of datasets tailored for different types of graph algorithms. For general graph convolutional networks (GCN) and graph attention networks (GAT), it provides support for popular datasets such as 'cora_tiny', 'amazon', 'cora_full', and 'reddit', which are widely used in standard graph learning tasks. Additionally, for specialized models like relational GCN (RGCN) and relational GAT (RGAT) that handle heterogeneous graphs, YiTu includes support for datasets with complex relational structures, such as 'aifb_hetero', 'mutag_hetero', 'bgs_hetero', and 'am_hetero'. This comprehensive dataset compatibility enables YiTu to perform efficiently across various graph types, from homogenous to heterogeneous networks, supporting diverse applications in graph analytics.
```bash
PYTHONPATH=. python3 test/bench_macro.py --info
```
+ Expected output:
```bash
----------[aifb-hetero]----------
n_nodes: 7262
node-types: 7
meta-paths: 104
n_rows: 216138
n_edges: 48810
avg_degrees: 4.030543059612123
----------[mutag-hetero]----------
n_nodes: 27163
node-types: 5
meta-paths: 50
n_rows: 281757
n_edges: 148100
avg_degrees: 0.5861625145724975
...
```

### 4. YiTu_T

The four datasets are available to download from AWS S3 bucket using the `down.sh` script. The total download size is around 350GB.

To use your own dataset, you need to put the following files in the folder `\DATA\\<NameOfYourDataset>\`

1. `edges.csv`: The file that stores temporal edge informations. The csv should have the following columns with the header as `,src,dst,time,ext_roll` where each of the column refers to edge index (start with zero), source node index (start with zero), destination node index, time stamp, extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV should be sorted by time ascendingly.
2. `ext_full.npz`: The T-CSR representation of the temporal graph. We provide a script to generate this file from `edges.csv`. You can use the following command to use the script 
    >python gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): The torch tensor that stores the edge featrues row-wise with shape (num edges, dim edge features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
5. `labels.csv` (optional): The file contains node labels for dynamic node classification task. The csv should have the following columns with the header as `,node,time,label,ext_roll` where each of the column refers to node label index (start with zero), node index (start with zero), time stamp, node label, extrapolation roll (0 for training node labels, 1 for validation node labels, 2 for test node labels). The CSV should be sorted by time ascendingly.

## Running

### Without NDP

#### 1. YiTu_GNN

To run the sample programs, run the following on root directory:

```shell
python example/YiTu_GNN/single/gcn/main.py --data_path /home/data/planetoid/data --dataset citeseer --dim 3703 --classes 6
python example/YiTu_GNN/single/gcn/main.py --data_path /home/data/planetoid/data --dataset cora --dim 1433 --classes 7
python example/YiTu_GNN/single/gcn/main.py --data_path /home/data/planetoid/data --dataset pubmed --dim 500 --classes 3
```

#### 2. YiTu_GP

To run the sample programs, run the following on root/examples/YiTu_GNN/NDP directory:
```shell
nondp/bfs-w --input bcsrgraph_path --source 1
nondp/cc-w --input bcsrgraph_path --source 1
nondp/pr-w --input bcsrgraph_path
nondp/sssp-w --input bcsrgraph_path --source 1
nondp/bc-w --input bcsrgraph_path --source 1
```

#### 3. YiTu_H

To run the sample programs, run the following on root/YiTu_GNN/YiTu_H directory:
performance comparison with baseline in terms of throughput and memory consumption.
```bash
# R-GCN on AIFB
PYTHONPATH=. python3 test/bench_macro.py --lib=dgl --model=rgcn --dataset=aifb_hetero --d_hidden=32
PYTHONPATH=. python3 test/bench_macro.py --lib=XGNN_H --model=rgcn --dataset=aifb_hetero --d_hidden=32

# R-GAT on AIFB (slower)
PYTHONPATH=. python3 test/bench_macro.py --lib=dgl --model=rgat --dataset=aifb_hetero --d_hidden=32
PYTHONPATH=. python3 test/bench_macro.py --lib=XGNN_H --model=rgat --dataset=aifb_hetero --d_hidden=32
```
+ Expected output:
```bash
[DGL] aifb-hetero, DGLRGCNModel, d_hidden=32
allocated_bytes.all.allocated: 384.49 MB
allocated_bytes.small_pool.allocated: 330.37 MB
allocated_bytes.large_pool.allocated: 54.12 MB
throughput: 1.0x
...
[HGL] AIFBDataset, RGCNModel, d_hidden=32
allocated_bytes.all.allocated: 58.97 MB
allocated_bytes.small_pool.allocated: 45.17 MB
allocated_bytes.large_pool.allocated: 13.80 MB
throughput: 15.0x~20.0x
...
```
+ Benchmark Parameters:
  + --lib: `'dgl', 'XGNN_H', 'pyg'`
  + --model: `'gcn', 'gat', 'rgcn', 'rgat'`
  + --dataset: `'cora_tiny', 'amazon', 'cora_full', 'reddit'` (for `'gcn', 'gat'`), `'aifb_hetero', 'mutag_hetero', 'bgs_hetero', 'am_hetero'` (for `'rgcn', 'rgat'`)
  + --d_hidden: `32` is the recommanded hidden size

#### 4. YiTu_T

To run the sample programs, run the following on root/YiTu_GNN/YiTu_T directory:
##### Single GPU Link Prediction
>python train.py --data \<NameOfYourDataset> --config \<PathToConfigFile>

##### MultiGPU Link Prediction
>python -m torch.distributed.launch --nproc_per_node=\<NumberOfGPUs+1> train_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs>

##### Dynamic Node Classification

Currenlty, TGL only supports performing dynamic node classification using the dynamic node embedding generated in link prediction. 

For Single GPU models, directly run
>python train_node.py --data \<NameOfYourDATA> --config \<PathToConfigFile> --model \<PathToSavedModel>

For multi-GPU models, you need to first generate the dynamic node embedding
>python -m torch.distributed.launch --nproc_per_node=\<NumberOfGPUs+1> extract_node_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs> --model \<PathToSavedModel>

After generating the node embeding for multi-GPU models, run
>python train_node.py --data \<NameOfYourDATA> --model \<PathToSavedModel>

### With NDP(better to use NDP on large graph)

#### 1. YiTu_GNN

1. Running Graph Server
To run the graph server, run the following on root directory:

* NDP Store Server
```shell
python examples/YiTu_GNN/NDP/server/NDP_server.py --dataset /home/reddit --num-workers [1]
```

2. Running Trainer
To run the trainer, run the following on root/examples/YiTu_GNN/NDP directory:

* Graph Convolutional Network (GCN)

```shell
python demo.py --YiTu_GNN 1 --method gcn --dataset /home/reddit --gpu [0] --feat-size 602
```

* Graph Isomorphism Network (GIN)

```shell
python demo.py --YiTu_GNN 1 --method gin --dataset /home/reddit --gpu [0] --feat-size 602
```

#### 2. YiTu_GP
The application takes a graph as input as well as some optional arguments. For example:

```shell
python demo.py --YiTu_GNN 0 --method bfs --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method cc --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method pr --input bcsrgraph_path
python demo.py --YiTu_GNN 0 --method sssp --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method sswp --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method bc --input bcsrgraph_path --source 1
```

For applications that run on unweighted graphs and weighted graphs, the input argument are both the graph file (.bcsr). For weighted graphs, the edgeWeight file (.bcsrw) should be in the same directory as the graph file (.bcsr).

The source argument is an integer to indicate the source vertex, and the source vertex id is 0 By default.
