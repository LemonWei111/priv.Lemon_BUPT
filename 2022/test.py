from skimage import io
import matplotlib.pyplot as plt
pic = io.imread('white.jpg') / 255. # 加载图片并进行归一化处理
#io.imshow(pic)
w,h,d=pic.shape   # 查看图像数据形状信息，即：宽、高位128、128，3通道
# 类似之前的操作，重置图像大小
data = pic.reshape(w*h, d)
#data.shape
#导入k-means库
from sklearn.cluster import KMeans
# 构建kmeans算法模型
model = KMeans(n_clusters=160, n_init=100)
# 开始训练
model.fit(data)
# 得到各簇中心点
centroids = model.cluster_centers_  
#print(centroids.shape)              # 查看簇的形状
C = model.predict(data)
# 获取每条数据所属簇
#print(C.shape)
#wc=C.shape
#centroids[C].shape
# 使用kmeans算法得到的数据得到压缩后的图片
compressed_pic = centroids[C].reshape((w,h,d))
# 绘制原图和压缩图片
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
