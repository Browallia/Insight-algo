# InSight Algorithm
The algorith of InSight

## env requirements

- torch\==1.3.0+cpu / torch\==1.3.1(CUDA)
- torchvision\==0.4.1+cpu / torchvision==0.4.2(CUDA)

## Manual

训练：运行train.py	--args详见代码

**PS**

```
python train.py --data_dir "./data" --epochs 20 --save_dir "checkpoint1.pkl" 
```

预测：运行predict.py	--args详见代码

**PS**

```
python predict.py --save_dir "checkpoint1.pkl" --dirpic "./data/test/0/4478.jpg"
```

数据集：按照图片标签分类

## TODO

算法模型改进

数据集更新
