from sklearn import datasets
from numpy import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time
import matplotlib.pyplot as plt
#求欧氏距离（两矩阵之间）
def euclDistance(vector1,vector2):
    #np.power(array,m)表示对array的每个元素求m次方
    return sqrt(sum(power(vector2-vector1,2)))

def initCentroids(dataSet,k):
    #矩阵的行、列数
    numSamples,dim=dataSet.shape
    #创建k行、dim列的全0数组
    centroids=zeros((k,dim))
    for i in range(k):
        index=int(random.uniform(0,numSamples))
        #将dataSet中第index+1行赋值给centroids的第i+1行
        centroids[i,:]=dataSet[index,:]
    return centroids

def kmeans(dataSet,k):
    #读取有几个样本数据
    numSamples=dataSet.shape[0]
    #将数组转化为矩阵
    clusterAssment=mat(zeros((numSamples,2)))
    clusterChanged=True
    centroids=initCentroids(dataSet,k)
    while clusterChanged:
        clusterChanged=False
        #遍历整个样本全部数据
        for i in range(numSamples):
            #临时存到μk的最小距离
            minDist=10000.0
            minIndex=0
            #k个簇
            for j in range(k):
                distance=euclDistance(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist=distance
                    minIndex=j
            #只要样本所属类别有变化，就继续循环
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
                clusterAssment[i,:]=minIndex,minDist**2
        for j in range(k):
            pointsInCluster=dataSet[nonzero(clusterAssment[:,0].A==j)[0]]
            #当pointsInCluster长度不为0，计算标注为j的所有样本的平均值
            if len(pointsInCluster)!=0:
                centroids[j,:]=mean(pointsInCluster,axis=0)
    print("聚类完毕！")
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    numSamples,dim=dataSet.shape
    if dim!=2:
        print("数据有误")
        return 1
    #样本颜色
    mark=['or','ob','og','ok','^r','+r','sr','dr','<r','pr']
    if k>len(mark):
        print("k太大")
        return 1
    for i in range(numSamples):
        #为样本指定颜色
        markIndex=int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
    #中心色
    mark=['Dr','Db','Dg','Dk','^b','+b','sb','db','<b','pb']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
    plt.show()

x=1
while x:
    print("请问您想用自己存储的testdata文件还是鸢尾花数据？1：testdata，2:鸢尾花")
    m=int(input())
    while m!=1 and m!=2:
        print("输入有误，请重新输入")
        m=int(input())
    if m==2:
        iris=datasets.load_iris()
        X,Y=iris.data,iris.target
        dataSet=X[:,[1,3]]
    if m==1:
        dataSet=[]
        fileIn=open("./testdata.txt")
        for line in fileIn.readlines():
            temp=[]
            lineArr=line.strip().split('\t')
            temp.append(float(lineArr[0]))
            temp.append(float(lineArr[1]))
            dataSet.append(temp)
        fileIn.close()
    print("数据读取完毕，正在为您展示散点图，您可以参考并决定将数据分为几类")
    dataSet=mat(dataSet)
    numSamples,dim=dataSet.shape
    for i in range(numSamples):
        plt.plot(dataSet[i,0],dataSet[i,1],'or')
    plt.show()
    print("分为几类？")
    k=int(input())
    while k>13:
        print("k太大,请重新指定")
        k=int(input())
    centroids,clusterAssment=kmeans(dataSet,k)
    showCluster(dataSet,k,centroids,clusterAssment)
    print("是否继续聚类？1继续0退出")
    x=int(input())
    if x!=0 and x!=1:
        print("输入错误，已退出")
        x=0
print("感谢使用")
