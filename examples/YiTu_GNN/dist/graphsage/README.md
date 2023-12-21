# GraphSAGE with YiTu_GNN

## 运行示例

> 构建docker镜像时安装ssh并启动sshd，然后配置容器之间的免密登录，在容器内对图进行partition然后将partition数据拷贝到
其他服务器的docker容器中。

假设我们拥有`11.11.11.16`、`11.11.11.17`、`11.11.11.18`、`11.11.11.19`四台服务器，如下我们将在这四台服务器上配置运行环境并运行分布式的GraphSAGE示例程序。

1. 在每个服务器节点上启动一个名为dYiTu_GNN的docker容器

    ```shell
    docker run --gpus all --rm -dit --shm-size="5g" --network=host --name dYiTu_GNN YiTu_GNN /bin/bash
    ```
    * `--shm-size="5g"`：表示共享内存大小设置为5g
    * `--network=host`: 表示使用主机上的网络，即使用主机的IP和端口，不会虚拟IP地址和端口。

2. 安装ssh并启动ssh服务，注意：因为上面启动容器的时候使用的是主机网络，为了支持docker容器之间进行ssh连接，需要修改ssh的默认端口号以避免和主机的ssh服务发生冲突。

    ```shell
    # 进入容器dYiTu_GNN内部
    docker exec -it dYiTu_GNN bash
    # 1. 安装ssh
    apt-get update && apt-get install -y openssh-server

    # 2. 设置登录密码（默认为1234）并允许root使用密码登录
    echo 'root:1234' | chpasswd && \
        sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
        sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

    # 3. 将ssh的默认端口号修改为49253
    sed -i 's/\(^Port\)/#\1/' /etc/ssh/sshd_config && echo Port 49253 >> /etc/ssh/sshd_config
    
    # 4. 启动ssh服务
    service ssh start
    ```
3. docker容器间配置免密登录

    3.1. 在每个服务器的容器中执行如下命令生成密钥：
    ```shell
    ssh-keygen -t rsa
    ```
    3.2. 将本地公钥发送到其他容器，并追加到`~/.ssh/authorized_keys`文件中

    3.3. 配置ssh config文件，在`11.11.11.16`容器内配置其他容器的host信息，修改`~/.ssh/config`文件，添加如下内容（端口号就是我们修改的ssh端口号）。其他服务器上添加相同配置即可。
    ``` shell
    Host 11.11.11.16
        User root
        Port 49253
    Host 11.11.11.17
        User root
        Port 49253
    Host 11.11.11.18
        User root
        Port 49253
    Host 11.11.11.19
        User root
        Port 49253
    ```

4. 数据处理：对数据进行partition操作，因为有4个服务器，所以num_parts设置为4。partition之后将数据发送到其他服务器的相同目录中。
    4.1 首先将代码拷贝到每个docker容器`/home`目录中：
    ```shell
    docker cp YiTu_GNN dYiTu_GNN:/home
    ```
    4.2 在任一一个docker容器内执行partition，对于soc-LiveJournal1, soc-pokec-relationships数据集，首先需要利用`YiTu_GNN/datagen`中的脚本生成图特征和标签等信息：
    ```shell
    # partition
    cd /home/YiTu_GNN/examples/YiTu_GNN/dist/
    python partition_graph.py --dataset ogbn-products --num_parts 4 --output /home/data
    ```
    4.3 将partition数据发送到其他容器的相同目录中。

5. 分布式环境配置：将节点的IP（宿主机的IP）写入文件`ip_config.txt`文件中, 如下所示：
    ```
    11.11.11.16
    11.11.11.17
    11.11.11.18
    11.11.11.19
    ```
6. 执行程序：
    ```shell
   python /home/YiTu_GNN/YiTu_GNN/utils/dgl_launch.py --workspace /home/YiTu_GNN/examples/YiTu_GNN/dist/graphsage \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config /home/data/YiTu_GNN/ogbn-products/random/4part_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/opt/conda/bin/python main.py --graph_name ogbn-products --part_config /home/data/YiTu_GNN/ogbn-products/random/4part_data/ogbn-products.json --ip_config ip_config.txt --num_epochs 5 --eval_every 2 --num_hidden 16"
    ```
    * num_trainers: 表示每台机器训练器的数量
    * num_samplers: 表示给每个训练器分配的采样器进程数
    * num_servers: 表示每台机器服务端的数量
    * part_config: 表示分区的配置信息所在位置

## 相关问题

1. 程序挂起问题：在初始化进程组的时候可能由于一些设置不当导致程序挂起，可能是因为网卡设置不当，可以通过环境变量显示指定需要使用的网卡，如下表示使用网卡eno2。网卡信息可以通过`ifconfig`查看。
    ```python
    import os

    os.environ["NCCL_SOCKET_IFNAME"] = "eno2"
    ```

2. 输出日志：通过设置环境变量`NCCL_DEBUG`来设置nccl的日志输出级别：
    
    2.1 显示设置：
    ```python
    import os

    os.environ["NCCL_DEBUG"] = "INFO"
    ```
    2.2 通过参数设置额外的环境变量
    ```shell
    --extra_envs NCCL_DEBUG=INFO
    ```