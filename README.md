# LeNet
## 说明  
  本代码是在看完李沐老师的《动手深度学习Pytorch篇》+相应网络论文后，进行的网络复现；  
  欢迎各位使用并指出不足🫡  
## 环境依赖  
* python=3.8  
* torch=2.0.0  
* torchvision=0.15.0
## 目录结构
```
│  README.md          //帮助文档
│  LeNet.py  
├─data                //下载or存放数据集的文件 
│  │          
│  └─MNIST  
│      └─raw  
│              
├─LeNet               //存放结果的文件
│  └─runs  
│     └─acc.png       //训练结果
│     └─LeNet.pt      //训练结果
│     └─loss.png      //训练结果
```
## 使用说明  
1. 读取数据、构建网络等部分都放在了一个文件中（即LeNet.py）
2. 使用时只需要修改部分参数和配置
3. 默认使用MNIST数据集
4. 默认使用GPU进行训练
## 结果展示  
### 准确率 
![image](https://github.com/josieisjose/LeNet/blob/main/LeNet/runs/acc.png)
### 损失   
![image](https://github.com/josieisjose/LeNet/blob/main/LeNet/runs/loss.png)
