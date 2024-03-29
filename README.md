# YiTu_GNN & YiTu_GP

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