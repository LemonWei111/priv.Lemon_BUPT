#核心实现2-霍夫变换
'''
2024-03-17 v1
该版本实现基本圆形拟合，思想和问题：
1、对r也设置了格大小，一定程度上能减少计算量，
但太大时会导致拟合出的圆与实际圆偏离的情况严重；
2、时间复杂度大，依赖于edges非零点个数；
3、最大/最小半径选择依赖性高，适应能力和附近圆区分不强；

后续修订以以下格式注明：
########## 版本号 ##########

# 原代码
# 改动代码
# 添加代码

#说明

############################
'''
# import time
# st = time.time()

import cv2
import numpy as np
import edge_detection
from scipy import signal

########## 2024-03-30 v1.2 ##########
# 添加代码

# 说明：
# 决定是否打印中间调试输出，后续应用条件不再作说明

DEBUG_MODE = False #默认不打印
############################

# 参考cv2源码：
# https://blog.csdn.net/q13791565170/article/details/87940093?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171064798216800197030773%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171064798216800197030773&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-5-87940093-null-null.142^v99^pc_search_result_base6&utm_term=cv2.HoughCircles%28%29%E6%BA%90%E7%A0%81&spm=1018.2226.3001.4187
def hough_circle_detection(edges, min_dist, param1, param2, grid_size, min_radius, max_radius, gx, gy, angle):
    # 使用霍夫变换检测圆
    #:param edges 边缘检测后的图像
    #:param min_dist 检测到的圆之间的最小距离
    #:param param1 霍夫变换阈值，用于检测圆
    #:param param2 非极大值抑制阈值，用于筛选最高的局部值
    #:param grid_size 网格大小，用于霍夫空间的离散化
    #:param min_radius 圆的最小半径
    #:param max_radius 圆的最大半径
    #:param gx 图像的x方向梯度
    #:param gy 图像的y方向梯度
    #:param angle 图像的梯度方向
    #:return 检测到的圆的列表，每个圆由(x, y, r)表示，其中(x, y)是圆心坐标，r是半径

    # 参数初始化
    height, width = edges.shape
    '''
    if dp < 1:
        dp = 1
    '''
    if min_radius == None:
        min_radius = 1
    if max_radius == None:
        max_radius = int(np.sqrt(height**2 + width**2)) // 2
    radius_range = range(min_radius, max_radius + 1)
   
    # 初始化霍夫空间（默认圆心只能在图片上）
    '''
    # 创建一个哈希表来存储累加器的值
    accumulator = {}
    '''
    hough_space = np.zeros((height // grid_size[0] + 2, width // grid_size[1] + 2, (max_radius - min_radius) // grid_size[2] + 2), dtype=np.uint8)
    # hough_space = np.zeros((height // grid_size[0] + 1, width // grid_size[1] + 1, (max_radius - min_radius) // grid_size[2] + 1), dtype=np.uint64)

    if DEBUG_MODE:
        print(height // grid_size[0])
        print(hough_space.shape) # 打印霍夫空间的形状
    
    # 计算霍夫空间
    for y in range(height):
        for x in range(width):
            # 如果边缘点存在且水平/垂直梯度至少有一方不为0
            # 梯度方向current_angle = angle[x, y]
            # 梯度值current_magnitude = magnitude[x, y]
            if edges[y, x]:
                gxx = gx[y]
                gyx = gy[y]
                if (gxx[x] != 0 or gyx[x] != 0):
                    # 遍历半径范围
                    # 在梯度的方向上位移，一次正，一次反，每次位移r += 1，直到r到最大
                    k1 = 0
                    current_angle = angle[y, x]
                    sx = np.cos(current_angle)
                    sy = np.sin(current_angle)
                    while k1 < 2:
                        # 初始一个位移的启动
                        # 位移量乘以最小半径，从而保证了所检测的圆的半径一定是大于最小半径
                        x1 = x + min_radius * sx;
                        y1 = y + min_radius * sy;
                        # 在梯度的方向上位移
                        # r <= max_radius保证了所检测的圆的半径一定是小于最大半径
                        r = min_radius
                        while r <= max_radius:
                            grid_r = int((r - min_radius) // grid_size[2])
                            #print(grid_r)
                            # 将计算出的a、b和r根据网格大小进行归一化
                            grid_a = int(x1 // grid_size[0])
                            grid_b = int(y1 // grid_size[1])

                            # 如果位移后的点超过了累加器矩阵的范围，则退出
                            if grid_a < 0 or grid_a >= hough_space.shape[1] or grid_b < 0 or grid_b >= hough_space.shape[0]:
                                break
                            # 在累加器的相应位置上加1
                            hough_space[grid_b, grid_a, grid_r] += 1
                            #print(grid_b, grid_a, grid_r)
                            x1 += sx
                            y1 += sy
                            r += 1
                        # 把位移量设置为反方向
                        sx = -sx
                        sy = -sy
                        k1 += 1
                '''
                for r in range(radius_range):
                    grid_r = int((r - min_radius) // grid_size[2])
                    print(grid_r)
                    for theta in range(0, 181, dp):  # 只计算0到180度，dp为步长
                        theta_rad = np.deg2rad(theta)  # 将角度转换为弧度
                        # 根据当前点(x, y)、半径r和角度theta计算霍夫变换参数a和b
                        a = int(x - r * np.cos(theta_rad))
                        b = int(y - r * np.sin(theta_rad))
                        
                        # 更新累加器的值
                        #key = (a, b, r)
                        #accumulator[key] = accumulator.get(key, 0) + 1
                        
                        # 将计算出的a、b和r根据网格大小进行归一化
                        grid_a = int(a // grid_size[0])
                        grid_b = int(b // grid_size[1])
                        # 确保(a, b, r)在霍夫空间的有效范围内
                        if (grid_a >= 0 and grid_a <= hough_space.shape[1] - 1 and grid_b >= 0 and grid_b <= hough_space.shape[0] - 1 and grid_r >= 0 and grid_r <= hough_space.shape[2] - 1):
                            hough_space[grid_b, grid_a, grid_r] += 1
                            # print("hough_space accumulator+1")
                        
                        # else:
                            # print(grid_a,grid_b,grid_r,end = "  ")
                '''
    circles = []
    '''
    print(accumulator)
    # 根据阈值和最小距离筛选圆
    for key, value in accumulator.items():
        if value > param1:
            # 检查圆心之间的最小距离
            if all(np.sqrt((key[0] - x)**2 + (key[1] - y)**2) > min_dist for x, y, _ in circles):
                circles.append(key)
    '''
    # 非极大值抑制
    # 遍历半径范围
    for grid_r in range(hough_space.shape[2]):
        for grid_y in range(hough_space.shape[0]):
            for grid_x in range(hough_space.shape[1]):
                # 如果霍夫空间在当前位置上的计数超过阈值param1
                if hough_space[grid_y, grid_x, grid_r] > param1:
                    # 检查是否是最高的局部值
                    if is_local_max(hough_space, grid_y, grid_x, grid_r, min_dist, param2):
                        # 将检测到的圆心位置添加到结果中，考虑网格大小
                        x = int((grid_x + 0.5) * grid_size[0])
                        y = int((grid_y + 0.5) * grid_size[1])
                        r = int((grid_r + 0.5) * grid_size[2] + min_radius)
                        circles.append((x, y, r))
    
    return circles

def is_local_max(hough_space, y, x, r, min_dist, param2):
    #:param hough_space 霍夫空间
    #:param y 霍夫空间内圆心的 y 坐标
    #:param x 霍夫空间内圆心的 x 坐标
    #:param r 霍夫空间内圆的半径
    #:param min_dist 最小圆心间距
    #:param param2 圆半径变化范围
    #:return 如果给定点是局部最大值，则返回 True；否则返回 False

    # 检查是否是最高的局部值（附近邻域内）
    '''
    if x - 1 >= 0:
        if hough_space[y, x - 1, r] > hough_space[y, x, r]:
            return False
    if y - 1 >= 0:
        if hough_space[y - 1, x, r] > hough_space[y, x, r]:
            return False
    if r - 1 >= 0:
        if hough_space[y, x, r - 1] > hough_space[y, x, r]:
            return False
    if x + 1 < hough_space.shape[1]:
        if hough_space[y, x + 1, r] > hough_space[y, x, r]:
            return False
    if y + 1 < hough_space.shape[0]:
        if hough_space[y + 1, x, r] > hough_space[y, x, r]:
            return False
    if r + 1 < hough_space.shape[2]:
        if hough_space[y, x, r + 1] > hough_space[y, x, r]:
            return False
    '''

    ########## 2024-03-22 v1.1 ##########
    # 添加代码

    # 说明：用于限定y_search、y_search遍历范围，减少后续判断
    
    height, width, _ = hough_space.shape
    ############################
    
    min_dist =min_dist // grid_size[2]
    param2 = param2 // grid_size[2]
    # 遍历以(r, y, x)为中心的立方体内的所有点
    for r_search in range(r - param2, r + param2 + 1):
        # 确保r_search在霍夫空间的有效范围内
        if r_search >= 0 and r_search < hough_space.shape[2]:
            ########## 2024-03-22 v1.1 ##########
            # 原代码
            '''
            for y_search in range(y - min_dist, y + min_dist + 1):
                for x_search in range(x - min_dist, x + min_dist + 1):
                    # 确保(x_search, y_search)在霍夫空间的有效范围内
                    # 检查是否有比(hough_space[y, x, r])更大的值在立方体内部分
                    if (x_search >= 0 and x_search < hough_space.shape[1] and
                        y_search >= 0 and y_search < hough_space.shape[0] and
                        hough_space[y_search, x_search, r_search] > hough_space[y, x, r]):
                        # 如果找到更大的值，则不是局部最大值，返回False
                        return False
            '''
            # 改动代码

            # 说明：
            # 1、最初遍历即确保(x_search, y_search)在(hough_space[y, x, r])立方体内
            # 2、添加条件再次判断原始图像中是否相邻

            for y_search in range(max(0, y - min_dist), min(height, y + min_dist + 1)):
                for x_search in range(max(0, x - min_dist), min(width, x + min_dist + 1)):
                    if (hough_space[y_search, x_search, r_search] > hough_space[y, x, r] and
                        ((y_search - y)*grid_size[1])**2 + ((x_search - x)*grid_size[0])**2 < ((min_dist + 1)*grid_size[2])**2):
                        # 如果找到更大的值，则不是局部最大值，返回False
                        return False
            ############################
    
    if DEBUG_MODE:
        print("remain", "y", y, "x", x, "r", r)

    # 如果没有找到更大的值，则是局部最大值，返回True
    return True

def detect_coins(image_path, min_radius=0, max_radius=60, threshold=20, min_distance=60, param2=20, grid_size = [0, 0, 0]):
    #:param image_path 输入图像路径
    #:param min_radius 圆的最小半径
    #:param max_radius 圆的最大半径
    #:param threshold 霍夫变换阈值，用于检测圆
    #:param param2 非极大值抑制阈值，用于筛选最高的局部值
    #:param min_distance 检测到的圆之间的最小距离
    #:param grid_size 网格大小，用于霍夫空间的离散化
    #:return 检测到的圆的列表，每个圆由(x, y, r)表示，其中(x, y)是圆心坐标，r是半径

    edges, gx, gy, angle = edge_detection.edge_detection(image_path)

    if DEBUG_MODE:
        print('ok')
        print(edges.shape) # (1353, 1441)

    if grid_size == [0, 0, 0]:
        grid_size[:] = [edges.shape[0] // 50, edges.shape[1] // 50, 20]
    if grid_size == [0, 0, 1]:
        grid_size[:] = [edges.shape[0] // 20, edges.shape[1] // 20, 20]

    # min_dist是圆心之间的最小距离，param1是累加器的阈值，param2是圆的半径变化范围
    circles = hough_circle_detection(edges, min_dist=min_distance, param1=threshold, param2=param2, grid_size = grid_size, min_radius = min_radius, max_radius = max_radius, gx = gx, gy = gy, angle = angle)

    if DEBUG_MODE:
        print(circles)

    return circles

def visualize_circles(image_path, circles):
    # 可视化结果
    #:param image_path 输入图像路径
    #:param circles 圆

    # print(f"time:{time.time()-st}")
    print(f"检测到{len(circles)}个硬币")

    image = edge_detection.re_image(image_path)

    # output = np.zeros_like(edges)
    for x, y, r in circles:
        #cv2.circle(output, (x, y), r, 255, 1)
        cv2.circle(image, (x, y), r, 255, 1)
    cv2.namedWindow("Test_with_photo", cv2.WINDOW_NORMAL)
    # cv2.imshow("Test",output)
    cv2.imshow("Test_with_photo",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#### param ####
# 图像路径
image_path = './test3.jpg'

# 设置霍夫变换参数
min_radius = 0
max_radius = 30
threshold = 20  # 累加器阈值
min_distance = 60  # 局部圆最小距离
param2 = 20  #圆的半径变化范围
grid_size = [0, 0, 1]

#### main ####
print("WELCOME! This is just a test")
image_path, min_radius, max_radius, threshold, min_distance, param2, grid_size = './test2.jpg', 30, 50, 50, 100, 20, [0, 0, 0]

circles = detect_coins(image_path, min_radius, max_radius, threshold, min_distance, param2, grid_size)
visualize_circles(image_path, circles)
