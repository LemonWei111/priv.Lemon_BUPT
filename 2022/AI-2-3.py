import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#100行2列
n_data = torch.ones(100, 2)
#平均值、标准偏差
x0 = torch.normal(2*n_data, 1)
#1行100列
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
#100行1列
y1 = torch.ones(100)
#在给定维度0上对输入张量序列x0、x1进行拼接(inputs,dim)形成tensor
#默认生成32位浮点数
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
#64位整型
y = torch.cat((y0, y1), ).type(torch.LongTensor)
#使用Module类来自定义模型
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        #
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x
#定义网络、优化器与损失函数
net = Net(n_feature=2, n_hidden=10, n_output=2)
#print(net)
#优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
#画动态图
plt.ion()
for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        plt.cla()
        #输出out每行最大值
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(0.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        #绘图延时
        plt.pause(0.5)
#动态结束
plt.ioff()
plt.show()
