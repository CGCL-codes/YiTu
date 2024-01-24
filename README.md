# YiTu_GNN & YiTu_GP

## 安装

提供两种安装方式：
1. 利用Docker构建镜像
2. 利用conda安装环境

### 构建Docker镜像

1. 首先利用Docker构建YiTu_GNN运行的虚拟环境，Docker的安装请[参考](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)。如下我们构建了一个叫做`YiTu_GNN`的镜像，之后的所有实验都是在该镜像上完成的。
```shell
cd YiTu_GNN/docker
docker build -f Dockerfile -t YiTu_GNN .
```
2. 启动docker容器：
```shell
docker run --gpus all -e NCCL_SOCKET_IFNAME=eno1 --rm -dit --shm-size="5g" --network=host --name YiTu_GNN YiTu_GNN /bin/bash
```
* --gpus all：表示使用所有gpu
* -e NCCL_SOCKET_IFNAME=eno1: 设置需要使用的网卡
* --shm-size="5g": 设置共享内存大小
* --network=host: 使用主机的网络

3. docker容器内安装YiTu_GNN：
```shell
# 1. 将YiTu_GNN复制到docker
docker cp YiTu_GNN YiTu_GNN:/home
# 2. 进入容器
docker exec -it YiTu_GNN bash
cd /home/YiTu_GNN
# 3. 安装YiTu_GNN
python setup.py install
```

### 基于conda安装

1. 安装cmake:
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
2. 安装conda:
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
3. 安装相关依赖包：
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
# 单机依赖包
apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libgoogle-perftools-dev \
    protobuf-compiler && \
	rm -rf /var/lib/apt/lists/*
```
4. 编译安装pytorch:
```shell
mkdir ~/temp
cd ~/temp
# 下载能够支持多版本参数的PyTorch源码
git clone --recursive https://github.com/Ningsir/pytorch.git -b multi-version
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# 编译安装PyTorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
USE_NINJA=OFF python setup.py install --cmake

cd ~
rm -rf ~/temp
```

> 可能出现的问题：报错缺少valgrind.h文件
>
> 解决办法：cd third_party/valgrind && git checkout VALGRIND_3_18_0
5. 安装dgl:
```shell
conda install -y -c dglteam dgl-cuda10.2=0.7.1
```
6. 安装YiTu_GNN
```shell
python setup.py install
```

## 编译（需要Python 3.7）

进入根目录下的 examples/YiTu_GNN/NDP 目录

```shell
#编译有ndp的图算法
cmake .
make
#编译无ndp的图算法
cd nondp
make
cd ..
#编译图格式转换工具
cd tools
make
cd ..
```

## 数据预处理

### 1. YiTu_GNN

#### 不使用NDP

##### 下载数据集：

```shell
cd /home/data
git clone https://github.com/kimiyoung/planetoid.git
```

##### 数据预处理

```shell
python example/YiTu_GNN/single/data_process.py --data_path /home/data/planetoid/data --dataset citeseer
python example/YiTu_GNN/single/data_process.py --data_path /home/data/planetoid/data --dataset cora
python example/YiTu_GNN/single/data_process.py --data_path /home/data/planetoid/data --dataset pubmed
```

#### 使用NDP

##### 数据集准备
* 数据集格式 (`vnum`表示顶点个数)
  
  * `adj.npz`: 图邻接矩阵，大小为`(vnum, vnum)`，保存为`scipy.sparse`coo matrix 格式。
  * `labels.npy`: vertex label 大小为`(vnum,)`，保存为`numpy.array`格式。
  * `test.npy`, `train.npy`, `val.npy`: boolean array 大小为`(vnum,)`，保存为 `numpy.array`。
  * `feat.npy`(可选): feature of vertex 大小为`(vnum, feat-size)`，保存为 `numpy.array`. 若为提供，则可以随机生成。

* 数据集下载和格式转换
  
  ```shell
  mkdir reddit
  curl --output reddit/reddit.zip https://data.dgl.ai/dataset/reddit.zip
  unzip -d reddit reddit/reddit.zip
  python examples/YiTu_GNN/NDP/data/dgl2NDP.py --dataset reddit --out-dir /home/reddit
  ```

* 生成子图
  
  * hash partition
    
    ```shell
    python YiTu_GNN/NDP/partition/hash.py --num-hops 2 --partition 1 --dataset /home/reddit
    ```
  * dg-based partition
    
    ```shell
    python YiTu_GNN/NDP/partition/dg.py --num-hops 2 --partition 1 --dataset /home/reddit
    ```

### 2. YiTu_GP

#### 数据格式说明

.el 格式：每行两个整数，代表一条边的源节点和目的节点
```
0 1
0 3
2 3
1 2
```

.wel 格式：每行三个整数，代表一条边的源节点、目的节点、边权值
```
0 1 26
0 3 33
2 3 40
1 2 10
```

#### 数据格式转换

YiTu_GP 读取二进制的 CSR 图格式，这样的格式读取速度更快，且节省空间。转换方法如下：

进入根目录下的 examples/YiTu_GNN/NDP 目录
```
#.el 文件转成 .bcsr 文件
tools/converter path_to_Graph.el

#.wel 文件转成 .bcsr 文件 + .bcsrw 文件
tools/converter path_to_Graph.wel
```

## 运行

### 不使用NDP

#### 1. YiTu_GNN

##### 运行示例程序：

```shell
python example/YiTu_GNN/single/gcn/main.py --data_path /home/data/planetoid/data --dataset citeseer --dim 3703 --classes 6
python example/YiTu_GNN/single/gcn/main.py --data_path /home/data/planetoid/data --dataset cora --dim 1433 --classes 7
python example/YiTu_GNN/single/gcn/main.py --data_path /home/data/planetoid/data --dataset pubmed --dim 500 --classes 3
```

#### 2. YiTu_GP

进入根目录下的 examples/YiTu_GNN/NDP 目录
```shell
nondp/bfs-w --input bcsrgraph_path --source 1
nondp/cc-w --input bcsrgraph_path --source 1
nondp/pr-w --input bcsrgraph_path
nondp/sssp-w --input bcsrgraph_path --source 1
nondp/bc-w --input bcsrgraph_path --source 1
```

### 使用NDP(倾向大图使用)

#### 1. YiTu_GNN

1. Graph Server运行

* NDP Store Server
```shell
python examples/YiTu_GNN/NDP/server/NDP_server.py --dataset /home/reddit --num-workers [1]
```

2. Trainer运行
进入根目录下的 examples/YiTu_GNN/NDP 目录

* Graph Convolutional Network (GCN)

```shell
python demo.py --YiTu_GNN 1 --method gcn --dataset /home/reddit --gpu [0] --feat-size 602
```

* Graph Isomorphism Network (GIN)

```shell
python demo.py --YiTu_GNN 1 --method gin --dataset /home/reddit --gpu [0] --feat-size 602
```

#### 2. YiTu_GP

```shell
python demo.py --YiTu_GNN 0 --method bfs --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method cc --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method pr --input bcsrgraph_path
python demo.py --YiTu_GNN 0 --method sssp --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method sswp --input bcsrgraph_path --source 1
python demo.py --YiTu_GNN 0 --method bc --input bcsrgraph_path --source 1
```
## Contact
YiTu is developed by the HUST SCTS&CGCL Lab. If you have any questions, please contact Yu Zhang (zhyu@hust.edu.cn), Chuyue Ye (yechuyue@hust.edu.cn), Xin Ning (xinning@hust.edu.cn), Jian Cheng (jaycheng@hust.edu.cn), Chenze Lu (czlu@hust.edu.cn), Zhiying Huang (hzying@hust.edu.cn). We welcome you to commit your modification to support our project.