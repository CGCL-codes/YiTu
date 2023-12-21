# pytorch-1.7.0, dgl-cuda10.2 0.7.1
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
	apt-utils \
	build-essential \
	ca-certificates \
	ccache \
	sudo \
	wget \
	git \
	libjpeg-dev \
	libpng-dev \
	libnuma-dev \
    # 单机依赖包
    libboost-all-dev \
    libgoogle-perftools-dev \
    protobuf-compiler && \
	rm -rf /var/lib/apt/lists/*

# 安装cmake
COPY install/ubuntu_install_build.sh /install/ubuntu_install_build.sh
RUN bash /install/ubuntu_install_build.sh

# 安装conda
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh

ENV CONDA_ALWAYS_YES="true"

# 首先安装pytorch和dgl的一些依赖包,然后安装pytorch和dgl
RUN /opt/conda/bin/conda install -y python=3.7.5 astunparse numpy ninja pyyaml mkl \
	mkl-include setuptools cffi \
	typing_extensions future six \
	requests dataclasses \  
	pytest nose cython scipy \
	networkx matplotlib nltk \
	tqdm pandas scikit-learn && \
	/opt/conda/bin/conda install -y -c pytorch magma-cuda102 && \
	/opt/conda/bin/pip install ogb

# 编译安装pytorch
COPY install/ubuntu_install_pytorch.sh /install/ubuntu_install_pytorch.sh
RUN bash /install/ubuntu_install_pytorch.sh

# 安装dgl
RUN /opt/conda/bin/conda install -y -c dglteam dgl-cuda10.2=0.7.1

# 配置ssh登录
RUN echo 'root:1234' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

ENV CONDA_ALWAYS_YES=

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
