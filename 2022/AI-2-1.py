#为什么线性回归拟合不好？
#缺少预值，导致误差完全由y来调整
import numpy as np
import matplotlib.pyplot as plt
def sigmoid_derivative(s): 
    ds = s * (1 - s)
    return ds
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def relu_derivative(s):   
    s[s<=0] = 0
    s[s>0] = 1
    return s
def relu(x):
    return np.maximum(0,x)
# N batch_size，D_in 输入，H 隐藏层，D_out 输出
N, D_in, H, D_out = 20, 1, 64, 1
np.random.seed(0)
x = np.arange(0,N,1).reshape(N,D_in)*1.0
#randn函数返回一个或一组样本，具有标准正态分布
y = x + np.random.randn(N,D_out) 
w1 = np.random.randn(D_in, H) 
w2 = np.random.randn(H, D_out)
#加预值
b1=np.random.randn(H)
b2=np.random.randn(D_out)
learning_rate = 1e-5
plt.ion()
for t in range(20000):
    #把x、w1矩阵的乘积加上预值，赋值给h
    h = x.dot(w1)+b1
    h_relu = sigmoid(h)
    #h_relu = relu(h)
    #预值
    y_pred = h_relu.dot(w2)+b2
    #square()返回一个新数组，元素值为原数组元素的平方
    loss = np.square(y_pred - y).sum()
    # 进行反向传播，关键算法
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_b2 = np.ones(N).T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T) 
    grad_h = grad_h*sigmoid_derivative(h_relu)
    #grad_h = grad_h*relu_derivative(h_relu)
    grad_w1 = x.T.dot(grad_h)
    grad_b1 = np.ones(N).T.dot(grad_h)
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    b1 -= learning_rate * grad_b1
    b2 -= learning_rate * grad_b2
    if (t%1000==0):
        #matplotlib 维护的 figure 有数量上限，不断的创建新的 figure 实例，很容易造成内存泄漏
        #通过cla来清除当前figure中活动的axes
        plt.cla()
        #散点图
        plt.scatter(x,y)
        plt.scatter(x,y_pred)
        #绘制经过点的曲线
        plt.plot(x,y_pred,'r-',lw=1, label="plot figure")
        #给图中的点加标签
        plt.text(0.5,0, 't=%d:Loss=%.4f' % (t, loss), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)
plt.ioff()
plt.show()
