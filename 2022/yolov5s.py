#yolov5s输入端探索
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
import imageio.v2 as imageio
from imageio import imread
import pickle as pk
import random
import os
import math
import torch
def free(k,c,t):
    x=os.listdir('./images/')
    for i in range(len(x)):
            #img = imageio.imread('./images/')
            #files = [os. path.join('./images/', p) for p in sorted(os.listdir('./images/'))]
            # 打乱数据库中的图像顺序，随机获取一张图
        print("第",i,"张")
        im_name = x[i]
        im_path = os.path.join('./images',im_name)
        img=Image.open(im_path)
        w=img.width
        h=img.height
        #print(w,h)
        Img=img
        freecrop(w,h,Img,k,c,t)
    print("裁剪完毕")
def crop(x,y,Img,k,c,t):
    #图像裁剪（裁剪之后是灰色图像）
    #print("图像裁剪")
    #对于彩色图像,打开后，返回的图像对象的模式都是“RGB”
    #对于灰度图像,打开后，其模式为“L”
    #模式 L”为灰色图像，它的每个像素用 8 个 bit 表示，0 表示黑，255 表示白，其他数字表示不同的灰度。
    #将当前的图像转换为"L"模式
    #x1,y1,x2,y2
    #print("开始裁剪")
    box = (x,y,2*x,2*y)
    #图片预处理，这样裁剪的好处是可以使得box不会变形。
    #对原图进行扩展之后，以原图的box中心作为输出图的中心，按照[height，width]×crop_size/min_shape的大小进行裁剪
    region = Img.crop(box)
    #region = img_L.crop(box)
    if Img.mode == "F":
        Img = Img.convert('RGB')
    region.save("./imagess/test.png")
    listt=os.listdir('./imagess/')
    for item in listt:
        if item.endswith('.png'):
            src = os.path.join('./imagess/', item)
            dst = os.path.join('./imagess/',str(k))    
            try:
                os.rename(src, dst + '.png')
                #print ('converting %s to %s ...' % (src, dst))
                k = k + 1
            except:
                try:
                    dst = os.path.join('./imagess/',str(k)+str(k-10*c))
                    os.rename(src, dst + '.png')
                    k+=1
                    c+=1
                except:
                    try:
                        dst = os.path.join('./imagess/',str(k)+str(k-10*c)+str(k-10*c-100*t))
                        os.rename(src, dst + '.png')
                        k+=1
                        c+=1
                        t+=1
                    except:
                        continue
    #region.show()
'''
def change(W,H,w,h,img):
    #print("图片缩放")
    Nw=W/w
    Nh=H/h
    if Nw>Nh:
        N=Nh
    else:
        N=Nw
    #w=(1-N)*w/2
    #h=(1-N)*h/2
    img_array=np.array(img)
    #plt.imshow(img_array)
    #plt.show()
    Img= cv2.resize(img_array,dsize=None,fx=(1-N)/2,fy=(1-N)/2,interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite(Img)
    #Img= Img*255 
    #plt.figure()
    #plt.axis('off')  # 关闭坐标轴
    #plt.savefig(a.png, bbox_inches='tight', pad_inches=0)
    Img=Image.fromarray(Img)
    if Img.mode == "F":
        Img = Img.convert('RGB')
    #Img.show()
    #plt.imshow(Img)
    #plt.show()
    return Img
'''    
#Mosaic数据增强
#随机裁剪
def freecrop(w,h,Img,k,c,t):
    for j in range(int(math.sqrt(w+h))):
        x=np.random.randint(2,int(w/2)-1)
        y=np.random.randint(2,int(h/2)-1)
        #print(x,y)
        if abs(x-y)>2 :
            crop(x,y,Img,k,c,t)
            x=0
            y=0
'''
#随机缩放
def freechange(w,h,img):
    for i in range(int(math.sqrt(w+h))):
        print("第",i,"次随机缩放")
        W=np.random.randint(1,2*w)
        H=np.random.randint(1,2*h)
        #Img=change(W,H,w,h,img)
        freecrop(w,h,change(W,H,w,h,img))
        W=0
        H=0
'''
#自适应图片缩放
#change(800,600,w,h)

#自适应锚框计算
#parser.add arqument('--noautoanchor',action='store true',help='disable autoanchor check')

# 特征提取
def extract_features(image_path, vector_size=32):
    image = imageio.imread(image_path)
    try:
        # 可选的特征检测算法有 BRISK,AKAZE,KAZE 以及 ORB
        # 关键点检测
        alg = cv2.BRISK_create()
        # alg = cv2.AKAZE_create()
        # alg = cv2.KAZE_create()
        # alg = cv2.ORB_create() 
        kps = alg.detect(image)
        # 选取前 vector_size=32 个特征点
        # 特征点的个数取决于图像的大小以及颜色分布
        # 按照关键点响应值对特征点进行排序
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # 计算特征点上对应的特征向量
        kps, dsc = alg.compute(image, kps)
        # 将所有的特征向量组成一个大的特征值
        #
        dscc=np.array(dsc,'f')
        #
        dsc = dscc.flatten()
        # 预定义一个维度为 64*vector_size 的特征向量
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # 如果计算得到的特征向量小于预定义的大小，则在向量末尾补零
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return None
    return dsc
#数据存储
def batch_extractor(images_path, pickled_db_path="./features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    result = {}
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f) 
    # 将所有特征保存在 pickle 文件里
    with open(pickled_db_path, 'wb+') as fp:
        pk.dump(result, fp)
#图像特征匹配
class Matcher(object):
    def __init__(self, pickled_db_path="./features.pck"):
        with open(pickled_db_path,'rb+') as fp:
            self.data = pk.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix,dtype=object)
        self.names = np.array(self.names)

    def cdist(self, vector):
        # 计算图像之间的余弦距离
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)
        # 可选距离
        # chebyshev：切比雪夫距离
        # cityblock 街区距离
        # cosine：余弦夹角
        # mahalanobis：马氏距离
        # minkowski：闵可夫斯基距离 
        # euclidean：欧式距离
        # hamming：汉明距离
 
    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cdist(features)
        # 找到排名前 5 的匹配结果
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()

def show_img(path):
    img = imageio.imread(path)
    plt.imshow(img)
    plt.show()
 
def run():
    #图片数据库的位置
    images_path = './imagess/'
    files = [os. path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # 打乱数据库中的图像顺序，随机获取一张图
    sample = random.sample(files, 3)
    batch_extractor(images_path)
    ma = Matcher('./features.pck')
    # 查询图像名称
    s = './testt.jpg'
    # s = './images/test-1.jpg'
    #print ('Query image ==========================================')
    #show_img(s)
    names, match = ma.match(s, topn=3)
    #print ('Result images ========================================')
    for i in range(3):
        # 计算 cosine 距离，将相似性定义为 1-cosine 距离，当两个图像越近，相似值越高
        #print ('Match %s' % (1-match[i]))
        show_img(os.path.join(images_path, names[i]))

k=0
c=1
t=1
free(k,c,t)

run()
