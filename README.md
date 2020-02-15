# ResUnet-a
### 针对遥感影像的语义分割
## [English introduction here](https://github.com/mohuazheliu/ResUnet-a/edit/master/introduction/README.md)
### 论文地址[https://arxiv.org/abs/1904.00592]
### 原作者使用MXNet的实现[https://github.com/feevos/resuneta]
## 依赖
#### keras==2.2.2
#### tensorflow==1.9.0
#### cv2
#### numpy
## 训练
#### 修改train.py中unet.train的第一个参数为数据集地址，第二个参数为模型存放地址
#### 数据集文件格式见[https://github.com/mohuazheliu/ResUnet-a/blob/master/dataset-postdam/train/README.md]
## Predict
#### 参照test.py 使用model.predict对一张图片进行预测，使用model.visual可视化预测
### [Postdam数据集地址](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
#### 这边将图片压缩一倍后进行训练，训练损失和验证损失
![train accuracy](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/train_acc.png)
![train_loss](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/train_loss.png)
![val_accuracy](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/val_acc.png)
![val_loss](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/val_loss.png)
#### 测试结果
![1](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/12-true.png)![1](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/12-label.png)
### 预训练模型下载
#### [OneDrive-Postdam](https://1drv.ms/u/s!ApOgV5zmgyrmhwQsafmdwnxjD27m?e=46LRSq)
### [Paris数据集下载](https://zenodo.org/record/1154821#.XH6HtygzbIU)
#### 训练及验证损失
![train accuracy](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/train_acc_paris.png)
![train_loss](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/train_loss_paris.png)
![val_accuracy](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/val_acc_paris.png)
![val_loss](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/val_loss_paris.png)
#### 测试结果
![1](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/150-true.png)![1](https://github.com/mohuazheliu/ResUnet-a/blob/master/material/150-label.png)
### 预训练模型下载
#### [OneDrive-Paris](https://1drv.ms/u/s!ApOgV5zmgyrmhwcbPe_WhkUm9uZY?e=YnGYxX)

