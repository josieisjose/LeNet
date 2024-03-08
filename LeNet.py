import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optimizer
import torchvision
import argparse
import os
import matplotlib.pyplot as plt
#调整配置
#设备
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
#
parser=argparse.ArgumentParser(prog='LeNet')
parser.add_argument("--outputs_dir",type=str,default='LeNet/runs',help='the path of output dir')
parser.add_argument('--device',type=str,default=device,help='choose cpu or gpu')
parser.add_argument('--epoch',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--lr',type=float,default=0.05,help='learning rate')
parser.add_argument('--momentum',type=float,default=0.5)
parser.add_argument('--num_workers',type=int,default=0)
opt = parser.parse_args()
print(opt)
#创建输出文件
os.makedirs(opt.outputs_dir,exist_ok=True)
#加载数据集
train_data=torchvision.datasets.MNIST(root='./data',train=True,download=False,transform=torchvision.transforms.ToTensor())
val_data=torchvision.datasets.MNIST(root='./data',train=False,download=False,transform=torchvision.transforms.ToTensor())
train=DataLoader(dataset=train_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
val=DataLoader(dataset=val_data,batch_size=opt.batch_size,shuffle=False)
#定义为网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=5,padding=2)
        self.pool1=nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        self.pool2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.flat=nn.Flatten()
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=torch.sigmoid(self.conv1(x))
        x=self.pool1(x)
        x=torch.sigmoid(self.conv2(x))
        x=self.pool2(x)
        x=self.flat(x)
        x=self.fc1(x)
        x=torch.sigmoid(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)
        x=self.fc3(x)
        return x
#实例化网络
net=LeNet().to(device)
#定义损失函数
loss_func=nn.CrossEntropyLoss() #易错！！
#定义优化器
optim=optimizer.SGD(net.parameters(),lr=opt.lr,momentum=opt.momentum)
#开始训练
train_loss=[];val_loss=[]
train_acc=[];val_acc=[]
max_acc_train=0;min_loss_train=10
max_acc_val=0;min_loss_val=10
best_epoch=1
for epoch in range(opt.epoch):
    net.train()
    loss_total=0
    loss_test=0
    correct=0
    correct_test=0
    total_samples_t=0
    total_samples_v=0
    for i,(x,y) in enumerate(train,1):
        x=x.to(device)
        y=y.to(device)
        y_hat=net(x)
        loss=loss_func(y_hat,y)
        loss_total+=loss.data.sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        correct+=(torch.argmax(y_hat,dim=1)==y).sum().item()#重要！！
        total_samples_t+=y.size(0)
    print("-------------------train--------------------------")
    print(f'epoch:{epoch + 1},loss_avg:{loss_total/len(train):.3f},acc_rate:{correct/total_samples_t*100:.3f}%')
    train_loss.append(loss_total/len(train));train_acc.append(correct/total_samples_t*100)
    with torch.no_grad():
        for j,data in enumerate(val,1):
            x,y=data
            x = x.to(device)
            y = y.to(device)
            y_hat=net(x)
            loss=loss_func(y_hat,y)
            loss_test+=loss.data.sum()
            correct_test+=(torch.argmax(y_hat,dim=1)==y).sum()
            total_samples_v+=y.size(0)
    print("-------------------val----------------------------")
    print(f'epoch:{epoch + 1},loss_avg:{loss_test/len(val):.3f},acc_rate:{correct_test /total_samples_v* 100:.3f}%')
    val_loss.append(loss_test/len(val));val_acc.append(correct_test /total_samples_v* 100)
    if (loss_test/len(val))<min_loss_val and (correct_test / len(val)*100)*max_acc_val:
        max_acc_train=correct/len(train)*100
        min_loss_train=loss_total/len(train)
        max_acc_val = correct_test /len(val)*100
        min_loss_val =loss_test/len(val)
        best_epoch=epoch+1

#保存最终权重
torch.save(net.state_dict(),f'{opt.outputs_dir}/LeNet.pt')
#加载权重
net_copy=LeNet().to(device)
net_copy.load_state_dict(torch.load(f'{opt.outputs_dir}/LeNet.pt'))
#画loss图
fig=plt.figure()
ax=fig.subplots()
x=np.linspace(1,opt.epoch,opt.epoch)
y_loss_train=train_loss;y_loss_val=val_loss
y_loss_train = torch.tensor(y_loss_train)#重要！！
y_loss_val = torch.tensor(y_loss_val)#重要！！
y_loss_train=np.array(y_loss_train.cpu())
y_loss_val=np.array(y_loss_val.cpu())
plt.title("loss pic")
plt.ylabel("loss");plt.xlabel("epoch")
ax.plot(x,y_loss_train,c='red',label='train_loss')
ax.plot(x,y_loss_val,c='blue',label='val_loss')
ax.legend(loc='upper right')
plt.savefig(fname=f'{opt.outputs_dir}/loss.png')
plt.show()
#画acc图
fig=plt.figure()
ax=fig.subplots()
x=np.linspace(1,opt.epoch,opt.epoch)#（start,stop,num）
y_acc_train=train_acc;y_acc_val=val_acc
y_acc_train = torch.tensor(y_acc_train) #重要！！
y_acc_val = torch.tensor(y_acc_val)#重要！！
y_acc_train=np.array(y_acc_train.cpu())
y_acc_val=np.array(y_acc_val.cpu())
plt.title("acc pic")
plt.ylabel("acc/%");plt.xlabel("epoch")
ax.plot(x,y_acc_train,c='red',label='train_acc')
ax.plot(x,y_acc_val,c='blue',label='val_acc')
ax.legend(loc='upper right')
plt.savefig(fname=f'{opt.outputs_dir}/acc.png')
plt.show()
