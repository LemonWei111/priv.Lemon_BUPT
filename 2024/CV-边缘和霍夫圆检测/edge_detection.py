##核心实现1-边缘检测：
'''
2024-03-15 v1
该版本实现基本边缘检测，思想和问题：
1、高斯核创建强调sigma非零，指定基于核的大小给定合适值，
确保生成的高斯核足够平滑，同时又不至于过于模糊；
2、使用卷积近似替代偏导计算操作，卷积核先卷积，再与图片卷积，优化计算；
3、双门限的选择：使用原梯度的80%位数作为高门限，高门限的一半作为低门限，
解决门限设置不好把握大小的问题，实现自适应

后续修订以以下格式注明：
########## 版本号 ##########

# 原代码
# 改动代码
# 添加代码

#说明

############################
'''

import numpy as np
import math
import cv2
from scipy.signal import convolve2d

########## 2024-03-30 v1.2 ##########
# 添加代码

# 说明：
# 决定是否打印中间调试输出，后续应用条件不再作说明

DEBUG_MODE = False #默认不打印
############################

########## 2024-03-17 v1.1 ##########
# 添加代码

# 说明：
# 1、通过压缩图像来强化适合本任务（针对硬币图像的确定性、噪声）的边缘提取
# 2、同时服务于霍夫变换圆检测，减小运算量

def re_image(image_path, target_long_side = 128):
    #:param image_path 原始图像的文件路径
    #:param target_long_side 目标长边的大小（default:128）
    #:return 自适应缩放后的图像，为灰度图像

    # 读取原始图像，返回目标长边为target_long_side大小的自适应缩放图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 获取原始图像的尺寸
    height, width = image.shape[:2]

    if DEBUG_MODE:
        print(f"image.shape:{image.shape}")

    # 确定长边和短边
    if height > width:
        long_side = height
        short_side = width
    else:
        long_side = width
        short_side = height

    # 根据长边的大小调整短边的大小，保持长宽比
    if long_side == height:
        new_height = target_long_side
        new_width = int(width * (target_long_side / height))
    else:
        new_width = target_long_side
        new_height = int(height * (target_long_side / width))

    # 压缩图像
    compressed_image = cv2.resize(image, (new_width, new_height))
    '''
    # 显示压缩后的图像
    cv2.namedWindow("Compressed Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Compressed Image', compressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return compressed_image
############################

def gaussian_kernel(size = 5, sigma = 0):
    # 创建一个高斯核
    #:param size 高斯核大小（default:5）
    #:param sigma 高斯核标准差（default:0）
    #:return 高斯核

    '''
    kernel = np.zeros((size, size))
    center = size // 2
    if sigma == 0:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    sum = 0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum += kernel[i, j]
    return kernel / sum
    '''
    if sigma <= 0:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


def derivative_kernel(): # size):
    # 创建近似卷积核
    #:return x、y方向近似计算的偏导核

    kernel_x = np.float32([[0, 0, 0],
                         [0, -1, 1],
                         [0, 0, 0]])
    kernel_y = np.float32([[0, 1, 0],
                         [0, -1, 0],
                         [0, 0, 0]])
    '''
    # 创建一个用于计算x方向偏导的核
    kernel_x = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel_x[i, j] = i - (size // 2)
    '''
    return kernel_x, kernel_y

def combine_kernels(gaussian, derivative):
    # 将高斯核和偏导数核结合，得到高斯偏导数核
    #:param gaussian 高斯核
    #:param derivative 偏导核
    #:return 卷积核的卷积结果

    # 进行二维卷积操作
    result = convolve2d(gaussian, derivative, mode='same', boundary='fill', fillvalue=0)
    return result #填充边界

def c_img(image, kernel):
    # 二维卷积操作
    #:param image 输入图像
    #:param kernel 卷积核
    #:return 卷积结果

    if DEBUG_MODE:
        print(kernel.shape, kernel)
    '''
    output = np.zeros_like(image)
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    padded_image = np.pad(image, padding, 'constant')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
            #ValueError: operands could not be broadcast together with shapes (25,25) (25,9) 
    '''
    # 进行二维卷积操作(边界全零填充)
    output = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    return output

def gaussian_derivative(image, kernel_size1 = 5, sigma = 0):
    #:param image 输入图像
    #:param kernel_size1 用于生成高斯核的核大小（default:5）
    #:param sigma 高斯核的标准差（default:0——自适应计算）
    #:return 图像在 x 方向上的高斯偏导数，图像在 y 方向上的高斯偏导数

    # 计算高斯偏导数
    gaussian_kernel_ = gaussian_kernel(kernel_size1, sigma)
    derivative_kernel_x, derivative_kernel_y = derivative_kernel()
    combined_kernel_x = combine_kernels(gaussian_kernel_, derivative_kernel_x)
    combined_kernel_y = combine_kernels(gaussian_kernel_, derivative_kernel_y)
    return c_img(image, combined_kernel_x), c_img(image, combined_kernel_y)

def non_max_suppression(magnitude, angle):
    #:param magnitude 图像的梯度幅值
    #:param angle 图像的梯度方向
    #:return 执行非最大值抑制后的结果矩阵

    # 非最大值抑制
    rows, cols = magnitude.shape # 获取梯度幅值图像的行列数
    suppressed = np.zeros_like(magnitude) # 创建一个与梯度幅值图像大小相同的全零矩阵，用于存储非最大值抑制后的结果
    neighbor1 = 1000  # 设置默认值（设大一点）
    neighbor2 = 1000  # 设置默认值
    # 遍历梯度幅值图像的每个像素点
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_angle = angle[i, j] # 获取当前像素点的梯度方向
            # 根据梯度方向选择邻域
            if (0 <= current_angle < np.pi / 8) or (15 * np.pi / 8 <= current_angle <= 2 * np.pi):
                neighbor1 = magnitude[i, j+1]
                neighbor2 = magnitude[i, j-1]
                # print("neighbor", neighbor1, neighbor2)
            elif (np.pi / 8 <= current_angle < 3 * np.pi / 8) or (9 * np.pi / 8 <= current_angle < 11 * np.pi / 8):
                neighbor1 = magnitude[i-1, j+1]
                neighbor2 = magnitude[i+1, j-1]
            elif (3 * np.pi / 8 <= current_angle < 5 * np.pi / 8) or (11 * np.pi / 8 <= current_angle < 13 * np.pi / 8):
                neighbor1 = magnitude[i-1, j]
                neighbor2 = magnitude[i+1, j]
            elif (5 * np.pi / 8 <= current_angle < 7 * np.pi / 8) or (13 * np.pi / 8 <= current_angle < 15 * np.pi / 8):
                neighbor1 = magnitude[i-1, j-1]
                neighbor2 = magnitude[i+1, j+1]
            # 只有在当前点是局部最大值时才保留
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                suppressed[i, j] = magnitude[i, j]
                # print("remain")
    # 返回经过非最大值抑制处理后的结果矩阵
    return suppressed

def double_threshold(magnitude, suppressed):
    # 应用双门限（门限选择要合适，太大则没有一个点被保留）
    #:param magnitude 图像的梯度幅值
    #:param suppressed 执行非最大值抑制后的结果矩阵
    #:return 强边，弱边

    # 计算高阈值和低阈值
    high = np.percentile(magnitude, 80) # 梯度幅度的 80% 分位数，运算时在不改变原数据情况下进行排序并选取
    low = high * 0.5

    if DEBUG_MODE:
        print(f"high:{high}, low:{low}")

    strong = np.zeros_like(suppressed) # 创建与输入的抑制后图像大小相同的全零矩阵，用于存储强边缘的结果
    weak = np.zeros_like(suppressed) # 创建与输入的抑制后图像大小相同的全零矩阵，用于存储弱边缘的结果
    strong[suppressed >= high] = 1 # 大于等于高阈值的像素点标记为强边缘
    weak[(suppressed < high) & (suppressed >= low)] = 1 #介于高阈值和低阈值之间的像素点标记为弱边缘
    return strong, weak

def hysteresis(strong, weak):
    # 使用滞后阈值跟踪边缘
    # 思想：细边与粗边相连的可能是有效边，如果某个弱边缘像素点周围存在强边缘像素点，则将其标记为强边缘
    #:param strong 强边
    #:param weak 弱边
    #:return 更新后的强边缘图像

    rows, cols = strong.shape # 获取强边缘图像的行列数
    # 遍历强边缘图像的每个像素点
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if strong[i, j] == 1:
                continue
            # 如果当前像素点是弱边缘，则检查其周围8个邻域像素点
            # 如果有任何一个像素点是强边缘，则将当前像素点标记为强边缘
            if weak[i, j] == 1:
                if (strong[i-1, j-1] == 1 or strong[i-1, j] == 1 or strong[i-1, j+1] == 1 or
                    strong[i, j-1] == 1 or strong[i, j+1] == 1 or
                    strong[i+1, j-1] == 1 or strong[i+1, j] == 1 or strong[i+1, j+1] == 1):
                    strong[i, j] = 1
    # 返回更新后的边缘图像（更连续）
    return strong

def edge_detection(image_path):
    # 读取图像
    #:param image_path 图像路径
    #:return 最终边缘，原图像x、y方向偏导，图像的梯度方向
    
    ########## 2024-03-17 v1.1 ##########
    # 原代码
    #image = cv2.imread(image_path, 0)  # 读取图像为灰度
    # 改动代码
    
    # 说明：re_image函数调用，返回自适应压缩后的图像
    
    image = re_image(image_path)
    ############################
    
    if image is None:
        print("Could not read the image.")
        return

    # 计算高斯偏导数
    gx, gy = gaussian_derivative(image, kernel_size1 = 5, sigma = 0.6)

    if DEBUG_MODE:
        print('gaosi ok')
        print("gx", gx)
        print("gy", gy)

    # 计算梯度幅度和方向
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    if DEBUG_MODE:
        print('deta ok')
        print("magnitude", magnitude)
        print("angle", angle)

    # 非最大值抑制
    suppressed = non_max_suppression(magnitude, angle)

    if DEBUG_MODE:
        print('nms ok')

    # 应用双门限
    strong, weak = double_threshold(magnitude, suppressed) # 自适应高门限、低门限

    if DEBUG_MODE:
        print('the ok')

    # 使用滞后阈值跟踪边缘
    final_edges = hysteresis(strong, weak)

    if DEBUG_MODE:
        print('edge ok')
        print(np.all(final_edges == 0))
        print(np.count_nonzero(final_edges))
    
    # 可视化结果
    cv2.namedWindow("Edge Detection", cv2.WINDOW_NORMAL)
    cv2.imshow('Edge Detection', (final_edges * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_edges, gx, gy, angle

# 使用函数
# edge_detection('./test.jpg')
