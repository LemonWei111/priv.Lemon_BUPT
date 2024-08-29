#### test ####
import cv2
import numpy as np
import joblib
import os

model_path = "./svm_model.pkl"
visual_vocabulary_path = "./visual_vocabulary.pkl"
data_dir = "./15-Scene/"  # 数据目录

def find_image_path_by_filename(filename, data_dir):
    #:param filename：图片名称
    #:param data_dir：数据集路径
    #:return 图像完整路径

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == filename:
                return os.path.join(root, file)
    return None

from classify import bag_of_words_representation_v3, L

# 图像分类
def classify_image(image_path, visual_vocabulary, svm_model):
    #:param image_path：图像路径
    #:param visual_vocabulary：视觉词典
    #:param svm_model：分类器
    #:return 预测标签

    histogram = bag_of_words_representation_v3(image_path, visual_vocabulary, L)
    predicted_label = svm_model.predict([histogram])[0]
    '''
    raise NotFittedError(msg % {"name": type(estimator).__name__})
    sklearn.exceptions.NotFittedError: This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
    '''
    return predicted_label

if __name__ == "__main__":
    svm_model = joblib.load(model_path)
    visual_vocabulary = joblib.load(visual_vocabulary_path)

    mode = int(input("指定序号请输入0，使用自己的图片数据请输出1"))
    if mode == 1:
        while image_path:
            image_path = input("请输入图片路径xx.jpg，退出可直接换行")
            predicted_label = classify_image(image_path, visual_vocabulary, svm_model)
            print("Predicted label:", predicted_label)
    else:
        if mode != 0:
            print("非法输入，默认指定序号")
        num = int(input("指定分类图片序号1-4485,0退出"))
        while num:
            filename = f"{num}.jpg"  # 文件名

            image_path = find_image_path_by_filename(filename, data_dir)
            if image_path:
                print("Image path:", image_path)
                predicted_label = classify_image(image_path, visual_vocabulary, svm_model)
                print("Predicted label:", predicted_label)
                num = int(input("指定分类图片序号1-4485,0退出"))
            else:
                print(f"Image {filename} not found in directory {data_dir}")
                num = int(input("指定分类图片序号1-4485,0退出"))
    print("退出成功，感谢使用")

