import torch
import numpy as np
import matplotlib.pyplot as plt
N, D_in, H, D_out = 20, 1, 64, 1
np.random.seed(0)
#起点为0，终点为N，步长为1的排列，改为二维数组，返回数据元素的数据类型为float32
#float32在十进制中有8位,float\float64有16位
x = torch.tensor(np.arange(0,N,1).reshape(N,D_in),dtype=torch.float32)
#randn函数返回一个或一组样本，具有标准正态分布
y = x +torch.tensor(np.random.randn(N,D_out), dtype=torch.float32)
# 定义网络结构与损失函数
model = torch.nn.Sequential(
    #线性变换不改变输入矩阵x的行数，仅改变列数。
    #完成从输入层到隐藏层的线性变换
    torch.nn.Linear(D_in, H),
    #激活函数
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
#均方误差，不设置reduction，则默认mean（reduction可以设置为none/mean/sum）
loss_fn = torch.nn.MSELoss(reduction='sum')
#定义优化器，这里使用 Adam
learning_rate = 1e-3
#查看网络参数：model.parameters() 优化器的初始化，model.state_dict() 模型的保存
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
plt.ion()
for t in range(50000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 2000 == 0:
        plt.cla()
        #实现torch到numpy数据的转化，然后画散点图
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.scatter(x.data.numpy(),y_pred.data.numpy())
        plt.plot(x.data.numpy(),y_pred.data.numpy(),'r-',lw=1,label="plot figure")
        plt.text(0.5, 0, 't=%d:Loss=%.4f' % (t, loss), fontdict={'size': 20,'color':'red'})
        plt.pause(0.1)
    #将梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #更新参数
    optimizer.step()
plt.ioff()
plt.show()
