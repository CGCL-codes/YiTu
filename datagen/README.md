# 数据生成

为了将传统图计算中的数据集（比如：soc-LiveJournal1, soc-pokec-relationships）应用到gnn中，需要生成图特征和标签等信息。

```python
python datagen.py --data_path /home/data --dataset soc-LiveJournal1 --num_classes 11 --feat_dims 512
```
* data_path：数据集所在目录；
* dataset: 数据集名称；
* num_classes: 节点类数；
* feat_dims: 特征维度。
