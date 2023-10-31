from PIL import Image
from pylab import *
from PIL import ImageEnhance
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
#引入 Tesseract 程序,用于文本检测,以只读方式打开文件


im=Image.open("./test.jpg")
im_L = im.convert("L")

def crop():
    #图像裁剪（裁剪之后是灰色图像）
    print("图像裁剪")
    #对于彩色图像,打开后，返回的图像对象的模式都是“RGB”
    #对于灰度图像,打开后，其模式为“L”
    #模式 L”为灰色图像，它的每个像素用 8 个 bit 表示，0 表示黑，255 表示白，其他数字表示不同的灰度。
    #将当前的图像转换为"L"模式
    #x1,y1,x2,y2
    box = (200,8,300,90)
    #图片预处理，这样裁剪的好处是可以使得box不会变形。
    #对原图进行扩展之后，以原图的box中心作为输出图的中心，按照[height，width]×crop_size/min_shape的大小进行裁剪
    region = im_L.crop(box)
    #保存新的图片
    region.save("./crop_test.jpg")
    region.show()

def conbine():
    #图像合并，此处 1.jpg 与 2.jpg 所有通道必须有相同的尺寸
    #可以理解为图像重合？
    print("图像合并")
    im7 = Image.open("./7.jpg") 
    im9 = Image.open("./9.jpg")
    #print(im7.mode,im9.mode)
    #能提取出r , g , b 三个颜色通道的前提是被提取的图片的色彩模式为RGB(可以先查看图片的mode属性来看色彩模式)
    #分离RGB图片的3个颜色通道,实现颜色交换
    r1,g1,b1 = im7.split()
    r2,g2,b2 = im9.split()
    #print(r1.mode,r1.size,g1.mode,g1.size)
    #print(r2.mode,r2.size,g2.mode,g2.size)
    new_im=[r1,g2,b2]
    #print(len(new_im))
    #将n个通道合为一个彩色图像
    im_merge = Image.merge("RGB",new_im)
    im_merge.show()

def array():
    # 读取图像到数组中
    print("读取图像到数组中")
    imarray = np.array(im)
    imshow(imarray)
    x = [100,100,400,400]
    y = [20,50,20,50]
    # 使用红色星状标记绘制点
    plot(x,y,'r*')
    plot(x[:2],y[:2])
    #添加标题，显示绘制的图像
    title('Plotting: "test.jpg"')
    show()

def bright():
    #亮度增强
    print("亮度增强")
    enh_bri = ImageEnhance.Brightness(im)
    brightness = 1.5
    im_brightened = enh_bri.enhance(brightness)
    im_brightened.show()

def contrast():
    #对比度增强
    print("对比度增强")
    enh_con = ImageEnhance.Contrast(im)
    contrast = 1.5
    im_contrasted = enh_con.enhance(contrast)
    im_contrasted.show()

def sharp():
    #锐度增强
    print("锐度增强")
    enh_sha = ImageEnhance.Sharpness(im)
    sharpness = 3.0
    im_sharped = enh_sha.enhance(sharpness)
    im_sharped.show()

def onsee():
    #图像模糊
    print("图像模糊")
    im_blur = im.filter(ImageFilter.BLUR)
    im_blur.show()

def bar():
    #轮廓提取
    print("轮廓提取")
    im_contour = im.filter(ImageFilter.CONTOUR)
    im_contour.show()


def po():
    #画直方图
    print("画直方图")
    imarray=np.array(im_L)
    plt.figure("lena")
    #
    im_a=np.array(im,'f')
    #worry
    arr=im_a.flatten()
    n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75) 
    plt.show()

def cat():
    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
    threshold = 200
    '''
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    '''
    #
    fun=lambda x:0 if x<threshold else 1
    #worry
    # 图片二值化
    #二值图像：只有两种颜色，黑色和白色
    print("图片二值化")
    photo = im_L.point(fun,'1')
    photo.show()


def see():
    #验证码识别，只能识别标准文本框输入吗？
    print("验证码识别")
    image = Image.open("./9.jpg",mode='r')
    #print(image)   
    #识别图片文字
    print(pytesseract.image_to_string(image,lang='chi_sim'))


x=1
while x:
    print("0 退出")
    print("1 验证码识别示例")
    print("2 图像二值化示例")
    print("3 直方图示例")
    print("4 轮廓提取示例")
    print("5 图像模糊示例")
    print("6 锐度增强示例")
    print("7 对比度增强示例")
    print("8 亮度增强示例")
    print("9 图像读取到数组示例")
    print("10 图像合并示例")
    print("11 图像裁剪示例")
    x=int(input())
    while int(x)<0 or int(x)>11:
        print("输入有误，请重新指示")
        x=int(input())
    if x==1:
        see()
    if x==2:
        cat()
    if x==3:
        po()
    if x==4:
        bar()
    if x==5:
        onsee()
    if x==6:
        sharp()
    if x==7:
        contrast()
    if x==8:
        bright()
    if x==9:
        array()
    if x==10:
        conbine()
    if x==11:
        crop()
