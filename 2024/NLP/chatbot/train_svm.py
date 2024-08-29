import os
import jieba
import pandas as pd

from spa.classifiers import SVMClassifier

def pad_data(train_data):
    '''
    对嵌套列表（每个子列表代表一个样本的分词结果）进行填充，确保所有子列表长度一致。

    :param train_data (list[list[str]]): 二维列表，内部子列表为分词后的文本序列。

    :return corrected_padded_train_data (list[list[str]]): 填充后所有子列表长度一致的二维列表。
    '''
    # 寻找最长子列表的长度
    max_length = max(len(sublist) for sublist in train_data)

    # 下面修正为仅对短于max_length的子列表进行填充
    corrected_padded_train_data = []
    for sublist in train_data:
        padded_sublist = list(sublist) # 复制原子列表
        while len(padded_sublist) < max_length:
            padded_sublist.append("") # 在末尾添加填充字符
        corrected_padded_train_data.append(padded_sublist)

    return corrected_padded_train_data

def load_train_data(file):
    '''
    从CSV文件加载训练数据，进行分词处理，并对分词结果进行填充以确保所有样本长度相同。

    :param file (str): CSV文件名（不包含路径），该文件位于执行脚本的当前工作目录下。

    :return padded_train_data (list[list[str]]): 填充后的训练数据分词列表。
    :return train_labels (numpy.ndarray): 训练数据的标签数组。
    '''
    # 当前执行目录
    rootdir = os.getcwd()
    file_path = os.path.join(rootdir, file)
    print(file_path)
    
    # 加载数据
    data = pd.read_csv(file_path)

    train_datas = data.iloc[:, 1].values 
    train_labels = data.iloc[:, 0].values 

    train_data = []
    for user_input in train_datas:
        # 使用jieba分词
        words = [word for word in jieba.lcut(user_input)]
        train_data.append(words)
        
    return pad_data(train_data), train_labels

train_data, train_labels = load_train_data('spa/waimai_10k.csv')
print(train_data[0], train_labels[0])
svm = SVMClassifier(train_data, train_labels)
