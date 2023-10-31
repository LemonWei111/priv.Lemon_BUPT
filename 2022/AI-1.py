#本回归所用损失函数err=mean(yi-a*xi-b)
from sklearn.datasets import make_regression
#导入make_egession()
import matplotlib.pyplot as plt
#导入并重命名为plt
import numpy as np
#生成数据集，样本数、特征数、噪音、偏差
X,Y=make_regression(n_samples=100,n_features=1,noise=0.4,bias=50)

def plotLine(theta0,theta1,X,Y):
    #用来取出X中的最大值、最小值
    max_x=np.max(X)+100
    min_x=np.min(X)-100
    #在min和max中返回1000个等距样本
    xplot=np.linspace(min_x,max_x,1000)
    #x代入y=k*x+b
    yplot=theta0+theta1*xplot
    print("目前的参数 b=",theta0)
    print("目前的参数 k=",theta1)
    #画出线性模型（横坐标、纵坐标、颜色、标签）
    plt.plot(xplot,yplot,color='g',label='Regression Line')
    #画散点图
    plt.scatter(X,Y)
    #设置横坐标范围、纵坐标范围
    plt.axis([-10,10,0,200])
    #显示可视化图像
    plt.show()

def hypothesis(theta0,theta1,X):
    return theta0+theta1*X

def cost(theta0,theta1,X,Y):
    costValue=0
    for(xi,yi)in zip(X,Y):
        #最小二乘法算损失
        costValue+=0.5*((hypothesis(theta0,theta1,xi)-yi)**2)
    return costValue
#求导
def derivatives(theta0,theta1,X,Y):
    dtheta0=0
    dtheta1=0
    for(xi,yi)in zip(X,Y):
        dtheta0+=hypothesis(theta0,theta1,xi)-yi
        dtheta1+=(hypothesis(theta0,theta1,xi)-yi)*xi
    #取平均
    dtheta0/=len(X)
    dtheta1/=len(X)
    return dtheta0,dtheta1

def updateParameters(theta0,theta1,X,Y,alpha):
    #alpha表示学习率
    dtheta0,dtheta1=derivatives(theta0,theta1,X,Y)
    #参数更新
    theta0=theta0-(alpha*dtheta0)
    theta1=theta1-(alpha*dtheta1)
    return theta0,theta1

def LinearRegression(X,Y):
    theta0=np.random.rand()
    theta1=np.random.rand()
    #1000次参数更新，每100次更新打印
    for i in range(0,1000):
        if i%100==0:
            plotLine(theta0,theta1,X,Y)
        theta0,theta1=updateParameters(theta0,theta1,X,Y,0.005)

LinearRegression(X,Y)
#1-1
