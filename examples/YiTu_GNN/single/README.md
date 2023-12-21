# 单机系统

1. 下载数据集：
```
cd /home/data
git clone https://github.com/kimiyoung/planetoid.git
```

2. 数据预处理

```shell
python data_process.py --data_path /home/data/planetoid/data --dataset citeseer
python data_process.py --data_path /home/data/planetoid/data --dataset cora
python data_process.py --data_path /home/data/planetoid/data --dataset pubmed
```

3. 运行示例程序：
```
cd gcn
python main.py --data_path /home/data/planetoid/data --dataset citeseer --dim 3703 --classes 6
python main.py --data_path /home/data/planetoid/data --dataset cora --dim 1433 --classes 7
python main.py --data_path /home/data/planetoid/data --dataset pubmed --dim 500 --classes 3
```
