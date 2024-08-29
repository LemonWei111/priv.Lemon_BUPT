##nlp test1 SVD & SGNS：
'''
2024-03-31 v1
该版本实现基本基于两种方法的词向量训练和最终文本评估的计算和输出，思想和问题：
1、SVD前所有非零奇异值之和因数据太大或其他原因未实现监控和计算；
2、SGNS在epoch较大时出现梯度爆炸；
3、无法量化评估词向量在pku_sim_test.txt上的相似度准确性，
或许可以将计算得到的词向量相似度与人工标注（目前未提供）进行比较。

后续修订以以下格式注明：
########## 版本号 ##########

# 原代码
# 改动代码
# 添加代码

#说明

############################
'''
import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD

from scipy.sparse.linalg import svds
from scipy.sparse import lil_matrix
import SVD

import random
import mmap

file_path = './training.txt'
test_file_path = './pku_sim_test.txt'
output_file_path = './2021213513.txt'
epochs = 5

batch_size = 1000
embedding_dim = 100
min_word_frequency = 2
svd_vector_path = "./svd.txt"
sgns_vector_path = "./sgns.txt"

########## 2024-04-03 v1.1 ##########
# 添加代码
# 说明：
# 梯度爆炸优化方案的相关参数设置

CUT = True # bool
DECAY = False # bool 单独作用不大
INIT = 'xavier' # must be in ['mean', 'xavier']

lr_decay_factor = 0.8
lr_decay_epochs = 1
############################

########## 2024-04-10 v1.2 ##########
# 添加代码
# 说明：
# 调试信息显示的设置

DEBUG = False
############################

# 使用内存映射读取训练文件
def read_training_file(file_path):
    #:param file_path 语料数据的路径
    #:return 预处理后的以文本字符串为元素的语料数据列表

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    line = line.strip().decode('utf-8') # .split()  # 解码
                    # 过滤标点符号
                    chinese_punctuation = "、,，.。!！?？;；:：“”'‘'’(（)）[【]】<《>》——"
                    line = ''.join(char if char not in chinese_punctuation else ' ' for char in line)
                    data.append(line)
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)

    return data

#### test ####
# read_training_file(file_path)

def build_svd_embeddings(data, n_components = 5, min_word_frequency = 5):
    #:param data 预处理后的以文本字符串为元素的语料数据列表
    #:param n_components 维度（default:5）
    #:param min_word_frequency 最小词频（default:5）
    #:return 单词向量矩阵，词表，单词到索引的映射字典，索引到单词的映射字典

    # 构建词表
    # 拆分文本为单词
    words = [word for sentence in data for word in sentence.split()]
    # 统计单词频率
    word_counts = Counter(words)
    # 过滤出现次数小于min_word_frequency的词汇
    vocab = [word for word, count in word_counts.items() if count >= min_word_frequency]
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    '''
    # 构建共现矩阵
    X = np.zeros((len(data), len(vocab)), dtype=np.float32)
    print("X_ok")
    for i, line in enumerate(data):
        for word in line.split():
            # print(word)
            if word in vocab:
                X[i, word2idx[word]] += 1

    print(np.all(X == 0))
    '''
    # 稀疏矩阵+降数据精度缓解内存开销
    # 构建一个空的稀疏矩阵，行数为数据数量，列数为词汇表大小
    X_sparse = lil_matrix((len(data), len(vocab)), dtype=np.float32)

    for i, line in enumerate(data):
        for word in line.split():
            if word in vocab:
                X_sparse[i, word2idx[word]] += 1

    ########## 2024-04-10 v1.2 ##########
    # 原代码（k = 6，未实现全部奇异值数量统计和计算）
    '''
    # 计算矩阵的秩
    # rank_X = matrix_rank(X_sparse)
    
    # 对原始数据矩阵进行奇异值分解
    # U, s, Vt = np.linalg.svd(X, full_matrices = False)
    _, s, _ = svds(X_sparse) #, k = min(X_sparse.shape) - 1) # killed
    print(s)
    
    # 计算非零奇异值的数量、奇异值之和
    non_zero_singular_values = np.count_nonzero(s)
    all_singular_values_sum = np.sum(s)
    '''
    # 改动代码

    # 说明：
    # 计算非零奇异值数量、奇异值之和
    # （自定义的密集矩阵奇异值数量算法，不支持词典太大 min_word_frequency = 10）
    
    non_zero_singular_values, all_singular_values_sum = 15595, 120406 # SVD.svd(X_sparse)
    if DEBUG:
        print("non_zero_singular_values", non_zero_singular_values, "all_singular_values_sum", all_singular_values_sum)
        print("X_ok")
    ############################

    # 校正降维维度
    if n_components > non_zero_singular_values:
        n_components = non_zero_singular_values
        print(f"你的维度设置过大，已为您调整到允许设置的最大维度{non_zero_singular_values}")
    
    # 进行SVD分解
    svd = TruncatedSVD(n_components=n_components)
    
    svd.fit(X_sparse) #要转置，不然每一行表示的是一个维度
    word_vectors = svd.components_.transpose()

    if DEBUG:
        # 提取其他指标
        selected_singular_values_sum = np.sum(svd.singular_values_)
        singular_values_ratio = selected_singular_values_sum / all_singular_values_sum

        # 输出信息
        print("原始数据中非零奇异值的数量:", non_zero_singular_values)
        print("选取的奇异值数量:", n_components)

        print("选取的奇异值之和:", selected_singular_values_sum)
        print("全部奇异值之和:", all_singular_values_sum)
        print("奇异值选取比例:", singular_values_ratio)

        print(len(word_vectors), len(vocab), len(word2idx), len(idx2word))

    return word_vectors, vocab, word2idx, idx2word

def get_vector(word, word2idx, word_vectors):
    # 获取词向量
    #:param word 要获取词向量的单词
    #:param word2idx 单词到索引的映射字典
    #:param word_vectors 单词向量矩阵
    #:return 如果单词存在于词表中，则返回对应的词向量；否则返回None

    idx = word2idx.get(word)
    # print("idx", idx)
    if idx is None:
        # print(f"单词 '{word}' 不在词表中。")
        return None
    vector = word_vectors[idx]
    # print(f"词 '{word}' 的词向量：{vector}")

    return vector

'''
#### test ####
data = ["natural language processing and machine learning is fun and exciting",
        "deep learning is a subfield of machine learning",
        "word embeddings are dense vectors of words",
        "word embeddings are dense vectors of words",
        "word embeddings are dense vectors of words",
        "word embeddings are dense vectors of words",
        "word embeddings are dense vectors of words",
        "I love you baby",
        "I love you",
        "I love myself",
        "I love him, instead of you",
        "I study well",
        "Unless you have a good study method, can you learn well",
        "learning makes perfect",
        "I'm not going to learn this",
        "study makes perfect",
        "learning is good, through some study",
        "go and study it",
        "learning is good, through some study"]

word1 = 'learning'
word_vectors, vocab, word2idx, idx2word = build_svd_embeddings(data)
print("word_vectors", word_vectors)
# print(vocab)
print(word2idx)
# print(idx2word)
get_vector(word1, word2idx, word_vectors)
'''

# 计算余弦相似度
def cosine_similarity(v1, v2):
    #:param v1 第一个向量
    #:param v2 第二个向量
    #:return 余弦相似度

    if len(v1) != len(v2):
        raise ValueError("输入向量的维度不相同。")

    # 展平
    # v1 = v1.reshape(-1)
    # v2 = v2.reshape(-1)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        # raise ValueError("输入向量的范数为零，无法计算余弦相似度。")
        return 0

    similarity = dot_product / (norm_v1 * norm_v2)

    return similarity

def calculate_similarity(word1, word2, word2idx, word_vectors):
    #:param word1 第一个单词
    #:param word2 第二个单词
    #:param word2idx 单词到索引的映射字典
    #:param word_vectors 单词向量矩阵
    #:return 两个单词之间的相似度，如果其中一个单词不存在于词表中，则返回0

    vec1 = get_vector(word1, word2idx, word_vectors)
    if vec1 is None: # np.all(vec2 == 0):
        # print(f"单词 '{word1}' 不存在于词表中。")
        return 0
    vec2 = get_vector(word2, word2idx, word_vectors)
    if vec2 is None: # np.all(vec2 == 0):
        # print(f"单词 '{word2}' 不存在于词表中。")
        return 0
    similarity = cosine_similarity(vec1, vec2)

    return similarity

'''
#### test ####
word2 = 'study'
print(calculate_similarity(word1, word2, word2idx, word_vectors))
'''

class SkipGramNS:
    def __init__(self, data, embedding_dim = 100, window_size = 2, negative_samples = 5, learning_rate = 0.01, min_word_frequency = 5):
        #:param data 预处理后的以文本字符串为元素的语料数据列表
        #:param embedding_dim 词向量维数（default:100）
        #:param window_size 窗口大小（default:2）
        #:param negative_sample 负采样数（default:5）
        #:param learning_rate 训练算法的学习率（default:0.01）
        
        self.data = data
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        
        # 构建词汇表
        self.vocab = self.build_vocab(data, min_word_frequency)
        if not self.vocab:
            raise ValueError("词汇表为空，请提供更多的数据或降低最小词频阈值。")

        # 词汇表大小
        self.vocab_size = len(self.vocab)
        # 单词到索引的映射
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        # 索引到单词的映射
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        # 初始化词向量矩阵
        self.initialize_embeddings()
        
        
    def build_vocab(self, data, min_word_frequency = 5):
        #:param data 预处理后的以文本字符串为元素的语料数据列表
        #:param min_word_frequency 最小词频（default:5）
        #:return 词汇表列表

        if not data:
            raise ValueError("数据为空，请提供有效的语料数据。")

        # 拆分文本为单词
        words = [word for sentence in data for word in sentence.split()]
        # 统计单词频率
        word_counts = Counter(words)
        # 过滤出现次数小于min_word_frequency的词汇
        vocab = [word for word, count in word_counts.items() if count >= min_word_frequency]

        return vocab

    # 初始化输入和输出词向量矩阵
    def initialize_embeddings(self):
        ########## 2024-04-03 v1.1 ##########
        # 原代码
        # 均匀分布的随机数来生成初始的词向量
        #init_range = 0.5/self.embedding_dim
        # 改动代码

        # 说明：
        # 梯度爆炸解决方案3：调整初始化策略

        if INIT == 'mean':
            init_range = 0.5/self.embedding_dim
        elif INIT == 'xavier':
            # xavier初始化
            import math
            
            init_range = math.sqrt(6.0 / (self.vocab_size + self.embedding_dim))
        else:
            print("The way of initialize is wrong, use xavier instead.")
            init_range = math.sqrt(6.0 / (self.vocab_size + self.embedding_dim))
        ############################

        self.input_embeddings = np.random.uniform(-init_range, init_range, (self.vocab_size, self.embedding_dim))
        self.output_embeddings = np.random.uniform(-init_range, init_range, (self.vocab_size, self.embedding_dim))

    # 训练模型
    def train(self, epochs=5, batch_size=32, lr_decay_factor=0.9, lr_decay_epochs=3):
        #:param epochs 训练轮数（default:5）
        #:param batch_size 训练批次大小（default:32）

        if not self.data:
            raise ValueError("语料数据为空，请提供有效的语料数据。")

        if batch_size <= 0:
            raise ValueError("批次大小必须大于零。")

        for epoch in range(epochs):
            # 将语料库分成多个批次

            ########## 2024-04-03 v1.1 ##########
            # 添加代码
            # 说明：
            # 梯度爆炸解决方案2：应用学习率衰减

            if DECAY:
                # 每lr_decay_epochs个epoch结束时衰减学习率
                if epoch % lr_decay_epochs == 0 and epoch != 0:
                    self.learning_rate *= lr_decay_factor
            ############################
            
            # 保留所有 batch
            num_batches = len(self.data) // batch_size + (1 if len(self.data) % batch_size != 0 else 0)
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx = min(start_idx + batch_size, len(self.data))
                batch = self.data[start_idx:end_idx]    
            
            # 丢弃最后一个不满 size 的 batch
            # batches = [self.data[i:i+batch_size] for i in range(0, len(self.data), batch_size)]
            # for batch in batches:

                GRAD = False
                # 显示训练进度
                if j % 10 == 0:
                    # GRAD = True # 不监控梯度，则注释掉本行
                    print(f"Epoch {epoch+1}/{epochs}, Batch {j+1}/{num_batches}")

                for sentence in batch:
                    # 拆分句子为单词
                    words = sentence.split()
                    for i, target_word in enumerate(words):
                        # 获取目标词索引
                        target_idx = self.word2idx.get(target_word)
                        # 获取上下文单词
                        context_words = self.get_context(words, i)
                        for context_word in context_words:
                            # 获取上下文单词索引
                            context_idx = self.word2idx.get(context_word)
                            # 更新词向量
                            GRAD = self.update_embeddings(target_idx, context_idx, GRAD)

    # 获取上下文单词
    def get_context(self, words, idx):
        #:param words 单词列表
        #:param idx 目标单词的索引
        #:return 上下文单词列表

        start = max(0, idx - self.window_size)
        end = min(len(words), idx + self.window_size + 1)
        context_words = words[start:idx] + words[idx+1:end]

        return context_words

    # 更新词向量
    def update_embeddings(self, target_idx, context_idx, GRAD):
        #:param target_idx 目标词的索引
        #:param context_idx 上下文词的索引
        #:param GRAD 是否输出一次当前梯度
        #:return 不输出下一次梯度

        if target_idx is None or context_idx is None:
            return

        # 获取目标词向量
        target_embedding = self.input_embeddings[target_idx]
        # 获取上下文词向量
        context_embedding = self.output_embeddings[context_idx]

        # 获取负样本索引列表
        neg_samples = []
        while len(neg_samples) < self.negative_samples:
            # 从词汇表中随机选择负样本
            word = random.choice(self.vocab)
            if word != self.idx2word[target_idx] and word != self.idx2word[context_idx]:
                neg_samples.append(self.word2idx[word])

        for neg_idx in neg_samples:
            # 获取负样本词向量
            neg_embedding = self.output_embeddings[neg_idx]

            # 正样本得分
            pos_score = np.dot(target_embedding, context_embedding)
            # 负样本得分
            neg_score = np.dot(target_embedding, neg_embedding)
            
            # 正样本梯度
            pos_grad = self.sigmoid(neg_score) * context_embedding - (1 - self.sigmoid(pos_score)) * context_embedding
            # 负样本梯度
            neg_grad = (1 - self.sigmoid(neg_score)) * neg_embedding

            ########## 2024-04-03 v1.1 ##########
            # 添加代码
            # 说明：
            # 梯度爆炸解决方案1：监督梯度情况，执行梯度裁剪（监控过程代码不做额外标注）

            if GRAD:
                print(np.linalg.norm(pos_grad), np.linalg.norm(neg_grad))
                GRAD = False
            if CUT:
                max_grad_norm = 5.0  # 可以根据具体情况调整
                if np.linalg.norm(pos_grad) > max_grad_norm:
                    pos_grad *= max_grad_norm / np.linalg.norm(pos_grad)
                if np.linalg.norm(neg_grad) > max_grad_norm:
                    neg_grad *= max_grad_norm / np.linalg.norm(neg_grad)
            ############################

            # 更新目标词向量
            self.input_embeddings[target_idx] += self.learning_rate * pos_grad
            # 更新上下文词向量
            self.output_embeddings[context_idx] += self.learning_rate * pos_grad
            # 更新负样本词向量
            self.output_embeddings[neg_idx] += self.learning_rate * neg_grad

        return GRAD

    # Sigmoid函数
    def sigmoid(self, x):
        #:param x 输入值
        #:return sigmoid函数值
        
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)

'''
#### test ####
model = SkipGramNS(data)
model.train(epochs)

# 获取词向量
word = 'learning'
idx = model.word2idx.get(word)
vector = model.input_embeddings[idx]
print(f"词 '{word}' 的词向量：{vector}")
'''

# 使用内存映射输出测试结果
def create_test_file(test_file_path, output_file_path, svd_word2idx, svd_word_vectors, sgns_model):
    #:param test_file_path 测试词对数据的路径
    #:param output_file_path 输出相似度文件的路径

    try:
        with open(test_file_path, 'r', encoding='utf-8') as file:
            # 打开新文件进行写入
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    for line in iter(mm.readline, b""):
                        # 假设两个子词之间由空格分隔
                        # 去除行首尾的空白字符，例如空格、换行符
                        r_line = line.strip().decode('utf-8')
                        # 分割行字符串
                        subwords = r_line.split()

                        # 检查是否正好有两个子词
                        if len(subwords) == 2:
                            # 提取这两个子词
                            word1, word2 = subwords
                            # print(f"子词1: {word1}, 子词2: {word2}")
                        else:
                            print("这一行不包含两个子词。")

                        sim_svd = calculate_similarity(word1, word2, svd_word2idx, svd_word_vectors)
                        # print("svd_ok", end = ' ')
                        sim_sgns = calculate_similarity(word1, word2, sgns_model.word2idx, sgns_model.input_embeddings)
                        # print("sgns_ok")
                        
                        # 去除行尾的换行符
                        w_line = line.rstrip()
                        # 在行末添加制表符和数字
                        modified_line = f"{w_line}\t{sim_svd}\t{sim_sgns}\n"
                        # 写入新文件
                        outfile.write(modified_line)

    except FileNotFoundError:
        print("File not found:", test_file_path)
    
    except Exception as e:
        print("An error occurred:", e)
    
def load_vector(vector_path):
    #使用内存映射加载此表，加快速度
    # 加载保存好的词向量模型
    file = open(vector_path, 'r', encoding='utf-8')

    # 使用 mmap 将文件映射到内存中
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

    # 读取词向量文件的内容并解析
    vectors = {}
    for line in iter(mmapped_file.readline, b""):
        line = line.strip().split()
        word = line[0].decode('utf-8')
        vector = [float(x) for x in line[1:]]
        vectors[word] = vector

    # 关闭文件和内存映射
    mmapped_file.close()
    file.close()

    if DEBUG:
        # 查看词表大小
        print(len(vectors))
    return vectors

'''
#### test ####
svd_vectors = load_vector("./svd.txt")
'''

def calculate_similarity_v2(word1, word2, vectors):
    #:param word1 第一个单词
    #:param word2 第二个单词
    #:param vectors 词向量
    #:return 两个单词之间的相似度，如果其中一个单词不存在于词表中，则返回0
    try:
        vec1 = vectors[word1]
        vec2 = vectors[word2]
    except:
        return 0
    similarity = cosine_similarity(vec1, vec2)

    return similarity

'''
#### test ####
def c_v2_test(test_file_path, svd_vectors):
    with open(test_file_path, 'r', encoding='utf-8') as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b""):
                r_line = line.strip().decode('utf-8')
                subwords = r_line.split()
                if len(subwords) == 2:
                    word1, word2 = subwords
                    print(f"子词1: {word1}, 子词2: {word2}")
                else:
                    print("这一行不包含两个子词。")
                sim_svd = calculate_similarity_v2(word1, word2, svd_vectors)
                print(sim_svd, end = ' ')
c_v2_test(test_file_path, svd_vectors)
'''

def create_test_file_v2(test_file_path, output_file_path, svd_vector_path, sgns_vector_path):
    #:param test_file_path 测试词对数据的路径
    #:param output_file_path 输出相似度文件的路径
    svd_vectors = load_vector(svd_vector_path)
    sgns_vectors = load_vector(sgns_vector_path)

    try:
        with open(test_file_path, 'r', encoding='utf-8') as file:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    for line in iter(mm.readline, b""):
                        r_line = line.strip().decode('utf-8')
                        subwords = r_line.split()
                        if len(subwords) == 2:
                            word1, word2 = subwords
                            # print(f"子词1: {word1}, 子词2: {word2}")

                            sim_svd = calculate_similarity_v2(word1, word2, svd_vectors)
                            # print("svd_ok", end = ' ')
                            sim_sgns = calculate_similarity_v2(word1, word2, sgns_vectors)
                            # print("sgns_ok")
                        else:
                            # print("这一行不包含两个子词。")
                            sim_svd = 0
                            sim_sgns = 0

                        sim_svd = "{:.4f}".format(sim_svd)
                        sim_sgns = "{:.4f}".format(sim_sgns)
                        w_line = line.rstrip().decode('utf-8')
                        # print(w_line)
                        modified_line = f"{w_line}\t{sim_svd}\t{sim_sgns}\n"
                        outfile.write(modified_line)

    except FileNotFoundError:
        print("File not found:", test_file_path)
    
    except Exception as e:
        print("An error occurred:", e)
    
def write_vector2file(vector_path, idx2word, word_vectors):
    with open(vector_path, 'w', encoding='utf-8') as outfile:
        for idx in range(len(idx2word)):
            word = idx2word.get(idx)
            # print(word, idx)
            vector = word_vectors[idx]
            word_line = f"{word} "
            outfile.write(word_line)
            for v in vector:
                vector_line = f"{v} "
                outfile.write(vector_line)
            # 写入新文件
            outfile.write("\n")
            
#### main ####

## train ##
import time

data = read_training_file(file_path)
if DEBUG:
    print("data_ok")
    st = time.time()

## train_SVD ##
svd_word_vectors, _, svd_word2idx, svd_idx2word = build_svd_embeddings(data, n_components = embedding_dim, min_word_frequency = min_word_frequency)
write_vector2file(svd_vector_path, svd_idx2word, svd_word_vectors)
if DEBUG:
    svd = time.time()
    print("svd_ok")
    print(svd - st)

## train_SGNS ##
# svd = st
sgns_model = SkipGramNS(data, embedding_dim, min_word_frequency = min_word_frequency)

########## 2024-04-03 v1.1 ##########
# 原代码
#sgns_model.train(epochs, batch_size)
# 改动代码
# 说明：
# 引入学习率衰减的训练代码

sgns_model.train(epochs, batch_size, lr_decay_factor, lr_decay_epochs)
############################

write_vector2file(sgns_vector_path, sgns_model.idx2word, sgns_model.input_embeddings)
if DEBUG:
    sgns = time.time()
    print("sgns_ok")
    print(sgns - svd)

## output ##
# create_test_file(test_file_path, output_file_path, svd_word2idx, svd_word_vectors, sgns_model)
create_test_file_v2(test_file_path, output_file_path, svd_vector_path, sgns_vector_path)
# last_used: './svd_100dim.txt', './sgns_5_100dim_w.txt'
if DEBUG:
    print("output_ok")

