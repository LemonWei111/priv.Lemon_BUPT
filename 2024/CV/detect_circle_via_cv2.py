import time
st = time.time()

import cv2
import numpy as np

def detect_coins(image_path):
    # 读取图像
    image = cv2.imread(image_path, 0)
    if image is None:
        print("Could not read the image.")
        return

    # 应用高斯模糊以减少图像中的噪声
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 使用Canny边缘检测方法找到边缘
    edges = cv2.Canny(blurred, int(16.2217/128*530), int(32.4434/128*530), apertureSize=3)

    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.imshow("edges",(edges * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 使用霍夫变换检测圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=int(60/128*530),
                               param1=20, param2=int(20/128*530), minRadius=int(0/128*530), maxRadius=int(30/128*530))

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # 绘制圆心
            cv2.circle(image, (i[0], i[1]), 1, (0, 100, 100), 3)
            # 绘制圆轮廓
            cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 255), 3)
            print(i[0], i[1], i[2])

        cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
        cv2.imshow("Detect",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No coins detected.")

# 测试
image = cv2.imread('./test3.jpg')
# 使用函数
detect_coins('./test3.jpg')
print(f"time:{time.time()-st}")
