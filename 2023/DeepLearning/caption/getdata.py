# 数据处理与加载模块
# Version: 1.2
# Author: [魏靖]
#Change：1.0-1.1文本描述添加<start><end><pad>
#1.1-1.2数据集划分添加验证集（mktrainvaltest），默认从train中划分出0.1（约1000个数据）用作训练过程中的验证

# 导入必要的库
import os
import json
import random 
from PIL import Image
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
#单步调试时，若提示
#[nltk_data] Error loading punkt: <urlopen error [WinError 10061]
#[nltk_data]     由于目标计算机积极拒绝，无法连接。>
#或其他错误
#请先尝试重新运行本行
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#如果没有可以在数据集ClothesDataSet类中使用try1
#或者pip install torchvision(若你有多个python，只需"指定python的绝对路径\python.exe""你的pip地址\pip.exe")
#当然也可以用conda下载

# 用于对给定描述列表，得到一个词汇表
def word_index(texts):
    # 分解为单词序列，不过滤标点符号
    all_words = [word for text in texts for word in word_tokenize(text)]

    # 构建词汇表
    word_counts = Counter(all_words)
    word_index = {word: index + 1 for index, (word, _) in enumerate(word_counts.most_common())}
    
    # 添加开始符和结束符到词汇表的末尾
    word_index['<start>'] = len(word_index) + 1
    word_index['<end>'] = len(word_index) + 1
    word_index['<pad>'] = 0

    # 打印词汇表
    #print("Word Index:")
    #print(word_index)
    
    return word_index

#用于基于给定的词汇表，给出描述列表的整数映射序列和填充后的整数映射序列
def get_sequence(word_index, texts):
    # 将文本转换为整数序列
    sequences = [
        [word_index[word] for word in word_tokenize(text) if word in word_index]
        for text in texts
    ]
    
    # 在每个序列的开头添加开始符，在结尾添加结束符
    sequences = [[word_index['<start>']] + seq + [word_index['<end>']] for seq in sequences]

    # 填充序列，使它们具有相同的长度
    max_sequence_length = max(map(len, sequences))
    padded_sequences = [seq + [0] * (max_sequence_length - len(seq)) for seq in sequences]

    # 打印整数序列
    #print("\nSequences:")
    #print(sequences)

    # 打印填充后的序列
    #print("\nPadded Sequences:")
    #print(padded_sequences)
    
    return sequences, padded_sequences

#将指定路径path的json数据读取到data中并返回
def read_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# 获取词汇表，用于外部调用
def words(path):
    clothes = read_data(path)
    caption = [value.split(':')[-1].strip() for value in clothes.values()]
    all_words = word_index(caption)
    #print(all_words)
    return all_words

# 数据集类，用于加载图像和对应的文本描述
#输入路径path和图像处理函数transform
#返回训练图像tensor输入：image，文本描述序列tensor：caption，caplen为序列实际长度，如果需要您可以在后续使用
class ClothesDataSet(Dataset):

    def __init__(self, path, all_words, transform):
        super(ClothesDataSet, self).__init__()
        clothes = read_data(path)
        data_set = list(clothes.keys())
        self.data_set = data_set
        '''
        for file_name in list(self.train_clothes.keys()):
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)
        '''
        captions = [value.split(':')[-1].strip() for value in clothes.values()]
        caption_not_equal, caption_line = get_sequence(all_words, captions)
        self.caption_not_equal = caption_not_equal
        self.caption_line = caption_line
        # PyTorch图像预处理流程（若getitem中采用try1，请注释掉涉及到的参数transform）
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        #print(len(self.data_set))
        file_path = './deepfashion-multimodal/images/' + self.data_set[item]
        
        # 读取图像
        image = Image.open(file_path)
        #try2
        if self.transform is not None:
            image = self.transform(image)

        '''#try1
        # 转换为RGB格式
        image = image.convert("RGB")

        # 指定目标大小为512x512
        target_size = (512, 512)

        # 使用 PIL 库的 resize 方法将图像缩放到指定大小
        image = image.resize(target_size, Image.LANCZOS)
        #print(type(image))
        
        # TODO: 将image从numpy形式转换为torch.float32,并将其归一化为[0,1]
        # 转换为 PyTorch 的 Tensor，类型为 torch.float32
        torch_image = torch.from_numpy(np.array(image)).float()
        # 归一化到 [0, 1] 范围
        normalized_image = torch_image / 255.0
        # TODO: 用permute函数将tensor从HxWxC转换为CxHxW
        image = normalized_image.permute(2, 0, 1)
        '''
        #文本描述序列转化为long tensor类型
        caplen = len(self.caption_not_equal[item])
        #print(caplen)
        caption = torch.LongTensor(self.caption_line[item])
        return image, caption, caplen

#如果启用了dataloader中的pin_memory，就会让数据常驻内存，加快数据加载速度
#workers大，可以加快模型训练速度（上一个batch训练好之前下一个batch数据已经准备好），但同样内存开销大，也加重了CPU负担。
#num_workers的经验设置值是自己电脑/服务器的CPU核心数
#如果启用了to_cuda, 那么Dataloader不能启用pin_memory,也请将num_works保持默认值0

# 生成训练集和测试集的数据加载器
#划分train\test
def mktrainval(train_path, test_path, all_words, batch_size, workers=4):
    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #这里的ToTensor和Normalize完全没必要每读一次数据都处理一次，可以在数据加载到内存的时候就直接全部处理完
    #这样每个数据只需要经历一次ToTensor和Normalize，这会大大提高数据读取速度，
    train_set = ClothesDataSet(train_path, all_words, transform=tx)
    test_set = ClothesDataSet(test_path, all_words, transform=tx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader

# 生成训练集、验证集和测试集的数据加载器
#划分trainvaltest
def mktrainvaltest(train_path, test_path, all_words, batch_size, val_size=0.1, workers=4):
    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    clothes = read_data(train_path)
    data_set = list(clothes.keys())

    # 划分训练集和验证集
    train_data_set, val_data_set = train_test_split(data_set, test_size=val_size, random_state=42)

    train_set = ClothesDataSet(train_path, all_words, transform=tx)
    val_set = ClothesDataSet(train_path, all_words, transform=tx)  # 使用同一个路径，确保加载相同的数据集

    train_set.data_set = train_data_set
    val_set.data_set = val_data_set

    #print("Size of train_set:", len(train_set))
    #print("Size of val_set:", len(val_set))


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)

    test_set = ClothesDataSet(test_path, all_words, transform=tx)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)

    #print("Size of train_set:", len(train_set))
    #print("Size of val_set:", len(val_set))
    #print("Size of test_set:", len(test_set))
    
    return train_loader, val_loader, test_loader

'''
#外部调用时from getdata import words, mktrainval, mktrainvaltest
#test words:
train_path = './deepfashion-multimodal/train_captions.json'
all_words = words(train_path)
print(all_words)

#example:
texts = ["A man wears a short-sleeve.", "A man."]
a, b = get_sequence(all_words, texts)
print(a, b)

#test import:
test_path = './deepfashion-multimodal/test_captions.json'
batch_size = 1
#train_data_loader, test_data_loader = mktrainval(train_path, test_path, all_words, batch_size, workers=4)
train_data_loader, val_data_loader, test_data_loader = mktrainvaltest(train_path, test_path, all_words, batch_size, val_size=0.1, workers=4)
'''
