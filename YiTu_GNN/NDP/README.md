# NDP

## 系统环境

* Python 3
* PyTorch (v >= 1.3)
* DGL (v == 0.7.1)

## 数据集准备

* 数据集格式 (`vnum`表示顶点个数)
  
  * `adj.npz`: 图邻接矩阵，大小为`(vnum, vnum)`，保存为`scipy.sparse`coo matrix 格式。
  * `labels.npy`: vertex label 大小为`(vnum,)`，保存为`numpy.array`格式。
  * `test.npy`, `train.npy`, `val.npy`: boolean array 大小为`(vnum,)`，保存为 `numpy.array`。
  * `feat.npy`(可选): feature of vertex 大小为`(vnum, feat-size)`，保存为 `numpy.array`. 若为提供，则可以随机生成。
* 数据集下载和格式转换
  
  ```
  mkdir reddit
  curl --output reddit/reddit.zip https://data.dgl.ai/dataset/reddit.zip
  unzip -d reddit reddit/reddit.zip
  python NDP/data/dgl2NDP.py --dataset reddit --out-dir /folders/to/save
  ```
* 生成子图
  
  * hash partition
    
    ```
    python NDP/partition/hash.py --num-hops 2 --partition 2 --dataset xxx/datasetfolder
    ```
  * dg-based partition
    
    ```
    python NDP/partition/dg.py --num-hops 2 --partition 2 --dataset xxx/datasetfolder
    ```

## 运行

### Graph Server运行

* NDP Store Server

```
python server/NDP_server.py --dataset xxx/datasetfolder --num-workers [gpu-num]
```
### Trainer运行

* Graph Convolutional Network (GCN)

```
python examples/profile/NDP_gcn.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ',']
```

* Graph Isomorphism Network (GIN)

```
python examples/profile/NDP_gin.py --dataset xxx/datasetfolder --gpu [gpu indices, splitted by ','] 
```

