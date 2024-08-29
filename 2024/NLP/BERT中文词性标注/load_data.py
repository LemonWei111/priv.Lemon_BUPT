# 数据加载相关函数
# 导入必要的库
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import mmap

max_length = 128

def find_max_length(sequences):
    # 假设sequences是一个列表，其中包含多个列表
    max_length = 0
    for seq in sequences:
        if type(seq) is not list:
            max_length = len(sequences)
            break
        max_length = max(max_length, len(seq))
    return max_length

def find_min_length(sequences):
    # 假设sequences是一个列表，其中包含多个列表
    min_length = 10000
    for seq in sequences:
        if type(seq) is not list:
            min_length = len(sequences)
            break
        min_length = min(min_length, len(seq))
    return min_length

def pad_sequences(sequences, max_length, padding_value='<PAD>'):
    # 假设sequences是一个列表，其中包含多个列表
    # max_length = find_max_length(sequences)
    padded_sequences = []
    for seq in sequences:
        # 填充序列以达到max_length
        padded_seq = seq + [padding_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
    padded_sequences = torch.tensor(padded_sequences)
    return padded_sequences

def read_data_and_tags(file_path_data, file_path_tags):
    with open(file_path_data, 'r', encoding='utf-8') as f_data:
        with open(file_path_tags, 'r', encoding='utf-8') as f_tags:
            # 读取所有行
            lines_data = f_data.readlines()
            lines_tags = f_tags.readlines()
           
            # 确保数据和对齐
            assert len(lines_data) == len(lines_tags), "数据文件和标签文件的行数不匹配。"
           
            # 将每行数据和标签分割成字或词和对应的标签
            data = [list(line_data.strip().split()) for line_data in lines_data]
            tags = [line_tags.strip().split() for line_tags in lines_tags]
            '''
            ts = [tag for line in tags for tag in line]
            num_tags = len(set(ts))
            '''
            return data, tags # , num_tags

def read_data(file_path, max_length=128, padding_value='<PAD>'):
    data = []
    attention_mask = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    line = list(line.strip().split())
                    ll = len(line)
                    pad_num = max_length - ll
                    if pad_num >= 0:
                        line += pad_num * [padding_value]
                        data.append(line)
                        attention_mask.append([1] * ll + [0] * pad_num)
                    else:
                        start = 0
                        while start < ll:
                            end = start+max_length
                            if end > ll:
                                pad_num = max_length - ll + start
                                data.append(line[start:] + pad_num * [padding_value])
                                attention_mask.append([1] * (ll - start) + [0] * pad_num)
                                break
                            data.append(line[start:end])
                            attention_mask.append([1] * max_length)
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)

    return data, attention_mask

def read_tag_data(file_path, max_length=128, padding_value='O'):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    line = list(line.strip().split())
                    ll = len(line)
                    pad_num = max_length - ll
                    if pad_num >= 0:
                        line += pad_num * [padding_value]
                        data.append(line)
                    else:
                        start = 0
                        while start < ll:
                            end = start+max_length
                            if end > ll:
                                data.append(line[start:] + (max_length - ll + start) * [padding_value])
                                break
                            data.append(line[start:end])
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)

    return data

# 读取训练验证文本数据
def read_data_v2(file_path, length=128, padding_value='<PAD>'):
    '''
    从给定文件路径中读取文本数据，并按指定长度进行分割和填充。
    
    :param file_path: str 文件路径
    :param length: int, optional 分割长度，默认为128
    :param padding_value: str, optional 填充值，默认为'<PAD>'

    :return: 分割后的文本数据列表、注意力掩码列表
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().replace('\n', ' ').strip()  # 读取整个文件内容，并替换换行符为空格
            # print(content)
            elements = content.split()  # 按空格分割元素
        # 按长度l分割元素到列表中，如果最后一个子列表元素不足l，则用None填充
        result = [elements[i:i+length] for i in range(0, len(elements), length)]
        attention_mask = [[1]*length for i in range(0, len(elements), length)]
        if result and len(result[-1]) < length:
            result[-1].extend([padding_value] * (length - len(result[-1])))  # 用None填充最后一个子列表
            attention_mask[-1].extend([0] * (length - len(result[-1])))
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)
        
    return result, attention_mask

# 读取训练验证tag数据
def read_tag_data_v2(file_path, length=128, padding_value='O'):
    '''
    从给定文件路径中读取标签数据，并按指定长度进行分割和填充。
    
    :param file_path: str 文件路径
    :param length: int, optional 分割长度，默认为128
    :param padding_value: str, optional 填充值，默认为'O'

    :return: 分割后的标签数据列表
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().replace('\n', ' ').strip()  # 读取整个文件内容，并替换换行符为空格
            elements = content.split()  # 按空格分割元素
        # 按长度l分割元素到列表中，如果最后一个子列表元素不足l，则用None填充
        result = [elements[i:i+length] for i in range(0, len(elements), length)]
        if result and len(result[-1]) < length:
            result[-1].extend([padding_value] * (length - len(result[-1])))  # 用None填充最后一个子列表
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)
        
    return result

# 读取测试待标注数据
def read_test_data(file_path):
    '''
    从给定文件路径中读取测试待标注数据，并生成数据列表和对应的注意力掩码列表。
    
    :param file_path: str 文件路径

    :return: 测试数据列表、注意力掩码列表
    '''
    data = []
    attention_mask = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                line = list(line.strip().split())
                # print(line)
                # input()
                data.append(line)
                attention_mask.append([1] * len(line))
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)
    return data, attention_mask

# 定义一个函数来构建词汇表（for tag）
def build_vocab_v2(data):
    '''
    从给定的标注数据中构建词汇表，并生成字典形式的词汇表及其索引。
    
    :param data: list 包含标注数据的列表

    :return: 词汇表到索引的字典、索引到词汇表的字典、词汇表大小
    '''
    # 从数据中提取所有的字或词
    words = [word for line in data for word in line]
   
    # 创建一个不重复的词汇表
    vocab = set(words)
    num = len(vocab)
   
    # 将词汇表映射到索引
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word, num

# 定义一个函数来构建词汇表（for data，增加未识别词'<O>'）
def build_vocab(data):
    '''
    从给定的文本数据中构建词汇表，并生成字典形式的词汇表及其索引。
    
    :param data: list 包含文本数据的列表

    :return: 词汇表到索引的字典、索引到词汇表的字典、词汇表大小
    '''
    # 从数据中提取所有的字或词
    words = [word for line in data for word in line]
   
    # 创建一个不重复的词汇表
    vocab = set(words)
    vocab.add('<O>')
    num = len(vocab)
   
    # 将词汇表映射到索引
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word, num

# without pretrain and use own vocab
def get_idx(data, word2idx):
    '''
    将给定的数据转换为索引形式，使用自定义的词汇表进行索引映射。
    
    :param data: list 包含数据的列表
    :param word2idx: dict 词汇表到索引的映射字典

    :return: 包含数据索引的列表
    '''
    data_idx = []
    for line in data:
        line_data_idx = []
        for word in line:
            try:
                line_data_idx.append(word2idx[word])
            except:
                # print("not found")
                line_data_idx.append(word2idx['<O>'])
        data_idx.append(line_data_idx)
    return data_idx

# use pretrained
def get_idx_v2(data):
    '''
    使用BERT预训练模型对应的Tokenizer将给定的数据转换为索引形式。
    
    :param data: list 包含数据的列表
    :param word2idx: dict 词汇表到索引的映射字典

    :return: 包含数据索引的列表
    '''
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') # './huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f') # bert-base-chinese')
    data_idx = []
    for line in data:
        line_data_idx = tokenizer.convert_tokens_to_ids(line)
        # print(type(line_data_idx), line_data_idx)
        data_idx.append(line_data_idx)
    return data_idx

class CustomDataset(Dataset):
    def __init__(self, data_idx, tags):
        self.data_idx = data_idx
        self.tags = tags

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        input_ids = self.data_idx[idx]
        labels = self.tags[idx]
        return input_ids, labels

# 创建一个自定义的Dataset类
class CustomDataset_v2(Dataset):
    def __init__(self, data_idx, tags, mask):
        '''
        根据给定的数据索引、标签和注意力掩码创建一个自定义的数据集。
        
        :param data_idx: list 包含数据索引的列表
        :param tags: list 包含标签的列表
        :param mask: list 包含注意力掩码的列表
        '''
        self.data_idx = torch.tensor(data_idx)
        self.tags = torch.tensor(tags)
        self.attention_mask = torch.tensor(mask)
        print(self.data_idx.shape)
        print(self.tags.shape)
        print(self.attention_mask.shape)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        input_ids = self.data_idx[idx]
        labels = self.tags[idx]
        attention_mask = self.attention_mask[idx]
        return input_ids, labels, attention_mask

# 保存词表
def write_dict2file(path, word2idx):
    '''
    将词汇表写入文件。
    
    :param path: str 文件路径
    :param word2idx: dict 词汇表到索引的映射字典
    '''
    with open(path, 'w', encoding='utf-8') as outfile:
        for key, value in word2idx.items():
            word_line = f"{key} {value}"
            outfile.write(word_line)
            outfile.write("\n")

# 加载词表
def load_dict(path):
    '''
    从文件中加载词汇表。
    
    :param path: str 文件路径

    :return: 词汇表到索引的字典、词汇表大小
    '''
    # 使用内存映射加载此表，加快速度
    file = open(path, 'r', encoding='utf-8')
    # 使用 mmap 将文件映射到内存中
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    # 读取词向量文件的内容并解析
    data = {}
    for line in iter(mmapped_file.readline, b""):
        line = line.strip().split()
        idx = line[0].decode('utf-8')
        data[idx] = int(line[1])

    # 关闭文件和内存映射
    mmapped_file.close()
    file.close()

    return data, len(data)

'''
####  Use Example  ####
# 读取数据和标签
train_file_path_data = 'train.txt'
train_file_path_tags = 'train_TAG.txt'
dev_file_path_data = 'dev.txt'
dev_file_path_tags = 'dev_TAG.txt'

test_file_path_data = 'test.txt'

max_length = 16

train_data, train_mask = read_data_v2(train_file_path_data, max_length)
data_word2idx, _, _ = build_vocab_v2(train_data)
train_data_idx = get_idx(train_data, data_word2idx)
del train_data
write_dict2file("data_dict", data_word2idx)
del data_word2idx

train_tag = read_tag_data_v2(train_file_path_tas, max_length)
tag_word2idx, _, _ = build_vocab_v2(train_data)
train_tag_idx = get_idx(train_tag, tag_word2idx)
del train_tag
write_dict2file("tag_dict", tag_word2idx)
del tag_word2idx

# 创建DataLoader
batch_size = 32  # 批大小
data_loader = DataLoader(dataset=CustomDataset_v2(train_data_idx, train_tag_idx, train_mask), batch_size=batch_size, shuffle=True)
'''
