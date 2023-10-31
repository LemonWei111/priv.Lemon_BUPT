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

def kmeans(dataSet):
    k=2
    #print(len(dataSet))
    while k<13:
        #调试：开始第k-1次聚类
        print("开始第",k-1,"次聚类",sep="")
        numSamples=dataSet.shape[0]
        clusterAssment=mat(zeros((numSamples,2)))
        r=mat(zeros((numSamples,k)))
        clusterChanged=True
        centroids=initCentroids(dataSet,k)
        while clusterChanged:
            #设置迭代次数终止条件，避免无效k
            ss=1
            clusterChanged=False
            for i in range(numSamples):
                minDist=10000.0
                minIndex=0
                for j in range(k):
                    distance=euclDistance(centroids[j,:],dataSet[i,:])
                    if distance<minDist:
                        minDist=distance
                        minIndex=j
                        r[i,:]=0
                        r[i,j]=1
                    if clusterAssment[i,0]!=minIndex or ss==len(dataSet)**2:
                        clusterChanged=True
                        clusterAssment[i,:]=minIndex,minDist**2
            for j in range(k):
                #print("ok")
                pointsInCluster=dataSet[nonzero(clusterAssment[:,0].A==j)[0]]
                if len(pointsInCluster)!=0:
                    #print("ok")
                    centroids[j,:]=mean(pointsInCluster,axis=0)
            #print("ok")
            ss=ss+1
        #调试
        print("计算当前目标函数")
        for i in range(numSamples):
            for j in range(k):
                J=r[i,j]*((dataSet[i:]-centroids[j,:])**2)
        #第一次聚类完毕存一个goalJ的初始值
        if k==2:
            goalcentroids=centroids
            goalclusterAssment=clusterAssment
            goalk=k
            goalJ=J
        #如果当前聚类的J值小于goalJ了，则更新goalk
        if J<goalJ:
        #增加k-goalk行到原来的goalcentroids矩阵
            numSamples,dim=dataSet,shape
            b=mat(zeros((k-goalk,dim)))
            goalcentroids=row_stack((goalcentroids,b))
            goalcentroids=centroids
            goalclusterAssment=clusterAssment
            goalk=k
        #调试
        print("输出当前目标函数值：",goalJ,sep="")
        k=k+1
    print("聚类完毕！")
    return goalcentroids,goalclusterAssment,goalk

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

dataSet=[]
fileIn=open("./testdata.txt")
for line in fileIn.readlines():
    temp=[]
    lineArr=line.strip().split('\t')
    temp.append(float(lineArr[0]))
    temp.append(float(lineArr[1]))
    #向dataSet列表中添加元素
    dataSet.append(temp)
fileIn.close()
dataSet=mat(dataSet)
centroids,clusterAssment,k=kmeans(dataSet)
showCluster(dataSet,k,centroids,clusterAssment)
