#### test2. SIFT Kmeans Word-bag SVM ####
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import joblib
import random
from sklearn.metrics import accuracy_score, classification_report
import time

from sklearn.ensemble import AdaBoostClassifier

# 设置参数
num_clusters = 200 # K均值聚类中心数
kfold_splits = 5  # 交叉验证折数
visual_vocabulary_path = "./visual_vocabulary.pkl"  # vocablary保存路径
svm_model_save_path = "./svm_model.pkl"  # SVM模型保存路径
L = 2 #>=1

# 提取SIFT特征
def extract_sift_features(image_path):
    #:param image_path：图片路径
    #:return 整图SIFT描述符

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    '''
    # 可视化
    print(descriptors.shape) # (257, 128)
    print('特征描述符各维度之和', np.sum(descriptors)) # 未标准化
    if np.sum(descriptors) > 1:
    #     descriptors = descriptors/np.sum(descriptors)
    print('标准化特征描述符各维度之和', np.sum(descriptors)) # 标准化
    #### cite ####
    ## https://blog.csdn.net/qq_41112170/article/details/125849051?ops_request_misc=&request_id=&biz_id=102&utm_term=cv2.SIFT_create().detect%E5%87%BD%E6%95%B0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-7-125849051.142^v100^pc_search_result_base6&spm=1018.2226.3001.4187 ##
    # cv2.drawKeypoints可视化关键点。
    # 可以传入flags参数，选择绘制带有大小与方向的更为丰富的关键点信息。
    # img = cv2.drawKeypoints(gray, keypoints, img)
    img = cv2.drawKeypoints(gray, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ##############
    cv2.namedWindow("SIFT_photo", cv2.WINDOW_NORMAL)
    cv2.imshow("SIFT_photo",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return descriptors

# 获取所有图像的SIFT特征（随机抽一部分就好）
def get_all_sift_features(data_dir):
    #:param data_dir：数据集路径
    #:return 所有用于构建词表的SIFT特征数组

    all_features = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                features = extract_sift_features(image_path)
                all_features.extend(features)
    return np.array(all_features)

# 获取每个类别的图像及其数量
def get_images_per_class(data_dir):
    #:param data_dir：数据集路径
    #:return 类别-图像数哈希表

    images_per_class = {}
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            images_per_class[dir_name] = []
            dir_path = os.path.join(root, dir_name)
            image_files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]
            images_per_class[dir_name] = image_files
    return images_per_class

# 从每个类别中随机抽取指定数量的图像并提取其SIFT特征
def get_random_sift_features(data_dir, num_per_class=100):
    #:param data_dir：数据集路径
    #:param num_per_class：每类选取用于构建词表的图像数（default：100）
    #:return 所有用于构建词表的SIFT特征数组

    all_features = []
    images_per_class = get_images_per_class(data_dir)
    
    for class_name, images in images_per_class.items():
        if len(images) <= num_per_class:
            selected_images = images
        else:
            selected_images = random.sample(images, num_per_class)
        
        for image_name in selected_images:
            image_path = os.path.join(data_dir, class_name, image_name)
            features = extract_sift_features(image_path)  # 提取SIFT特征的函数
            all_features.extend(features)
    
    return np.array(all_features)

# 使用K均值聚类生成视觉词汇（最耗时→从训练集中随机抽取一部分子集来训练视觉词典）
def create_visual_vocabulary(all_features, num_clusters):
    #:param all_features：所有用于构建词表的SIFT特征数组
    #:param num_clusters：视觉词表词数
    #:return 视觉词表

    # kmeans = KMeans(n_clusters=num_clusters)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters) # 创建 Mini Batch K-Means 聚类器
    kmeans.fit(all_features)
    visual_vocabulary = kmeans.cluster_centers_
    return visual_vocabulary

# 使用词袋模型表示图像（直方图作为图像描述）
def bag_of_words_representation(image_path, visual_vocabulary):
    #:param image_path：图片路径
    #:param visual_vocabulary：视觉词表
    #:return 词表表示直方图

    features = extract_sift_features(image_path)
    histogram = np.zeros(num_clusters)
    if features is not None:
        for feature in features:
            feature = feature.reshape(1, -1)
            distances = np.linalg.norm(feature - visual_vocabulary, axis=1)
            closest_cluster_index = np.argmin(distances)
            histogram[closest_cluster_index] += 1
    return histogram

# 提取指定层级SIFT特征
def extract_sift_features_v2(image_path, layer_id):
    #:param image_path：图片路径
    #:param layer_id：层号(0..L-1)
    #:return 当前层SIFT描述符组

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    descriptors = []

    height, width = gray.shape[:2]
    num_divisions = 2**(layer_id)
    h_division_size = height // num_divisions
    w_division_size = width // num_divisions

    for y in range(0, height, h_division_size):
        for x in range(0, width, w_division_size):
            # print(x,y)
            if y+h_division_size>height or x+w_division_size>width:
                continue
            roi = gray[y:y+h_division_size, x:x+w_division_size]
            kp, des = sift.detectAndCompute(roi, None)
            descriptors.append(des)
            '''
            img = cv2.drawKeypoints(roi, kp, roi, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            ##############
            cv2.namedWindow("SIFT_photo", cv2.WINDOW_NORMAL)
            cv2.imshow("SIFT_photo",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
    # print(len(descriptors))
    return descriptors

def get_closest_cluster_indices(distances, k=5, like=50):
    #:param distances：距离
    #:param k：最小值选择数（default：5）
    #:param like：最小间隔距离（default：50）
    #:return 最近词组

    # 对距离值进行排序并获得索引
    sorted_indices = np.argsort(distances)
    closest_indices = [sorted_indices[0]]
    min_dis = distances[sorted_indices[0]]
    for i in range(1, k):
        if distances[sorted_indices[i]] - min_dis >= like:
            break
        closest_indices.append(sorted_indices[i])
    return closest_indices

# 加权分配
def bag_of_words_representation_v2(image_path, visual_vocabulary, L=2):
    #:param image_path：图片路径
    #:param visual_vocabulary：视觉词表
    #:param L：层数（default：2）
    #:return 词表表示直方图

    # 在不同层上提取SIFT特征（论文第二页）
    # Li=0,整图SIFT；Li=1,4个SIFT；Li=2,4**2个SIFT...
    # for part_photos, do next（for i in range(L):）
    histogram = np.zeros(num_clusters*((1-2**(2*L))//(1-2**2)))   
    j = 0
    for i in range(L):
        # print(i)
        x0 = 1/(2**L)
        xi = 1/(2**(L-i+1))
        part_features = extract_sift_features_v2(image_path, i)
        if part_features is None:
            continue
        for features in part_features:
            if features is None:
                continue
            # print(features.shape)
            for feature in features:
                feature = feature.reshape(1, -1)
                # 没有设置阈值，直接用的最近距离匹配
                distances = np.linalg.norm(feature - visual_vocabulary, axis=1)
                # 考虑：软分配：平均/按权分给最小的几个（最小的几个中，差值不大于阈值）
                '''
                print(distances)
                input()
                if min(distances) < 400:
                '''
                '''
                # 直接分配
                closest_cluster_index = np.argmin(distances)
                if i == 0:
                    histogram[closest_cluster_index+j] += x0
                else:
                    histogram[closest_cluster_index+j] += xi # Paper:wi=1/(2**(L-i))    My:wi=2(i+1)/((L+1)*(L+2))
                '''
                # 平均软分配
                close_cluster_indexs = get_closest_cluster_indices(distances)
                for index in close_cluster_indexs:
                    avg = 1/len(close_cluster_indexs)
                    if i == 0:
                        histogram[index+j] += avg*x0
                    else:
                        histogram[index+j] += avg*xi
                
            j += num_clusters
                
    #norm:
    s_h = np.sum(histogram)
    if s_h > 1:
         histogram = histogram/s_h # 标准化
    # print(len(histogram))
    return histogram

# 简单拼接
def bag_of_words_representation_v3(image_path, visual_vocabulary, L=2):
    #:param image_path：图片路径
    #:param visual_vocabulary：视觉词表
    #:param L：层数（default：2）
    #:return 词表表示直方图

    histogram = np.zeros(num_clusters*((1-2**(2*L))//(1-2**2)))   
    j = 0
    for i in range(L):
        part_features = extract_sift_features_v2(image_path, i)
        if part_features is None:
            continue
        for features in part_features:
            if features is None:
                continue
            for feature in features:
                feature = feature.reshape(1, -1)
                distances = np.linalg.norm(feature - visual_vocabulary, axis=1)
                close_cluster_indexs = get_closest_cluster_indices(distances)
                for index in close_cluster_indexs:
                    avg = 1/len(close_cluster_indexs)
                    histogram[index+j] += avg
            j += num_clusters
    s_h = np.sum(histogram)
    if s_h > 1:
         histogram = histogram/s_h # 标准化
    return histogram

# 准备数据集和标签
def prepare_dataset(data_dir, visual_vocabulary, L=2):
    #:param data_dir：数据集路径
    #:param visual_vocabulary：视觉词表
    #:param L：层数（default：2）
    #:return 词表表示直方图数组，标签数组

    data = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            label = int(dir)
            label_dir = os.path.join(root, dir)
            for file in os.listdir(label_dir):
                if file.endswith(".jpg"):
                    image_path = os.path.join(label_dir, file)
                    histogram = bag_of_words_representation_v3(image_path, visual_vocabulary, L)
                    data.append(histogram)
                    labels.append(label)
    return np.array(data), np.array(labels)

# 使用SVM分类器作为基模型使用提升算法进行图像分类//worse than svm
def image_classification(data, labels):
    #:param data：词表表示直方图数组
    #:param labels：标签数组
    #:return 最佳分类器，交叉验证准确率数组

    best_score = -1  # 初始化最佳准确率为负数
    scores = []
    best_clf = None  # 初始化最佳分类器为None

    kf = KFold(n_splits=kfold_splits, shuffle=True)
    for train_index, test_index in kf.split(data):
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
        # 创建AdaBoost-SVM分类器
        adaboost_svm = AdaBoostClassifier(base_estimator=clf, n_estimators=5, learning_rate=0.01, random_state=42)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # 训练模型
        adaboost_svm.fit(X_train, y_train)
        # 测试模型
        y_pred = adaboost_svm.predict(X_test)
        # 计算准确率
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_clf = adaboost_svm
        print("Accuracy:", score)
        class_report = classification_report(y_test, y_pred)
        print(class_report)
    return best_clf, scores

# 使用SVM分类器进行图像分类
def image_classification_v2(data, labels):
    #:param data：词表表示直方图数组
    #:param labels：标签数组
    #:return 最佳分类器，交叉验证准确率数组

    best_score = -1  # 初始化最佳准确率为负数
    scores = []
    best_clf = None  # 初始化最佳分类器为None

    kf = KFold(n_splits=kfold_splits, shuffle=True) # 随机划分
    for train_index, test_index in kf.split(data):
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced') # rbf/linear
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        # 测试模型
        y_pred = clf.predict(X_test)
        # 准确率
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_clf = clf
        print("Accuracy:", score)
        scores.append(score)
        # 分类报告（精确率、召回率、F1分数等指标）
        class_report = classification_report(y_test, y_pred)
        print(class_report)
    return best_clf, scores

def test(data_dir, svm, visual_vocabulary, num_per_class=60):
    #:param data_dir：数据集路径
    #:param svm：分类器
    #:param visual_vocabulary：视觉词表
    #:param num_per_class：每类选取测试图片数量

    data = []
    labels = []
    images_per_class = get_images_per_class(data_dir)
    
    for class_name, images in images_per_class.items():
        if len(images) <= num_per_class:
            selected_images = images
        else:
            selected_images = random.sample(images, num_per_class)
        
        for image_name in selected_images:
            image_path = os.path.join(data_dir, class_name, image_name)
            histogram = bag_of_words_representation_v2(image_path, visual_vocabulary, 1)
            data.append(histogram)
            labels.append(int(class_name)) # label = int(dir)
    X_test = np.array(data)
    y_test = np.array(labels)
    y_pred = svm.predict(X_test)
    print(y_pred)
    # 准确率
    score = accuracy_score(y_test, y_pred)
    print("Accuracy:", score)
    # 分类报告（精确率、召回率、F1分数等指标）
    class_report = classification_report(y_test, y_pred)
    print(class_report)

# 主函数
def main(data_dir): # , num_clusters):

    # 提取SIFT特征
    print("Extracting SIFT features...")
    # all_sift_features = get_all_sift_features(data_dir)
    random_sift_features = get_random_sift_features(data_dir)

    st = time.time()
    # 使用K均值聚类生成视觉词汇
    print("Creating visual vocabulary...")
    # visual_vocabulary = create_visual_vocabulary(all_sift_features, num_clusters)
    visual_vocabulary = create_visual_vocabulary(random_sift_features, num_clusters)
    print("Build vocab time", time.time()-st)
    
    # 保存词汇表
    print("Saving model parameters...")
    joblib.dump((visual_vocabulary), visual_vocabulary_path)
    print("Model parameters saved to", visual_vocabulary_path)

    # 加载已有词汇表
    # visual_vocabulary = joblib.load(visual_vocabulary_path)
    
    st = time.time()
    # 准备数据集和标签
    print("Preparing dataset...")
    data, labels = prepare_dataset(data_dir, visual_vocabulary, L)
    print("Prepare SOFT dataset time", time.time()-st)

    # 图像分类
    print("Performing image classification...")
    clf, scores = image_classification_v2(data, labels)

    # 输出结果
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", np.mean(scores))
    
    # 保存模型参数
    print("Saving model parameters...")
    joblib.dump(clf, svm_model_save_path)  # 保存SVM模型
    print("Model parameters saved to", svm_model_save_path)
    '''
    print("Radom test")
    # clf = joblib.load(svm_model_save_path)
    test(data_dir, clf, visual_vocabulary)
    '''
if __name__ == "__main__":
    data_dir = "./15-Scene/"
    print(num_clusters)
    main(data_dir) # , num_clusters)
