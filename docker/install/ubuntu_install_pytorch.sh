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
