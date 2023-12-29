# VGG19整体表示+GRU模型定义
# Version: 1.0
# Author: [魏靖]

#实现一个基于服装图像的描述生成模型；
#使用VGG19模型提取图像特征，然后通过GRU解码器生成描述文本；
#解码器使用了Beam Search算法来生成更准确的描述。

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence

# 定义图像编码器类
class ImageEncoder(nn.Module):
    def __init__(self, model):
        super(ImageEncoder, self).__init__()
        # 使用预训练的VGG19模型
        # 使用预训练的ResNet模型\GoogLeNet模型（InceptionV1）
        #self.model = model
        # # 移除分类器的最后一层全连接层
        del model.classifier[6] # (5)
        self.image_encoder = model
        #print(self.image_encoder)
        # GoogLeNet模型（InceptionV1）模型截取最后一个卷积层之前的部分作为图像编码器
        #self.image_encoder = nn.Sequential(*list(self.model.children())[:-2])


    def forward(self, image):
        # 将图像传递给图像编码器
        return self.image_encoder(image)
        #在后面再作通道位置转变

# 使用预训练的VGG19模型
#encodemodel = models.vgg19(pretrained=True)

# 获取vgg19模型的第一个Sequential, 也就是features部分.
#features = torch.nn.Sequential(*list(encodemodel.children())[0])

#print(encodemodel)
#print('features of vgg19: ', features[-3])
#print('vgg19: ', encodemodel)

# 使用预训练的ResNet模型（这里以ResNet50为例）
#encodemodel = models.resnet50(pretrained=True)
# 使用预训练的GoogLeNet（InceptionV1）模型
#encodemodel = models.googlenet(pretrained=True)
#image_encoder = ImageEncoder(encodemodel)
    
# 示例用法：
# 虚拟输入图像（用你的实际数据替换）
#image_input = torch.randn(32, 3, 224, 224)  # 批量大小为32，通道数为3，图像大小为224x224

# 获取图像编码
#encoded_image = image_encoder(image_input)

#(32, 3, 224, 224)
#vgg19
#([32, 512, 14, 14])
#resnet50
#([32, 2048, 1, 1])
#GoogLeNet模型（InceptionV1）
#([32, 1024, 1, 1])
# 打印编码后的图像形状
#print(encoded_image.shape)

# 定义解码器类
class Decoder(nn.Module):
    def __init__(self, image_code_dim, vocab_size, word_dim, hidden_size, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, word_dim)#vocab_size, word_dim)
        print("embed",vocab_size,word_dim)

        # 初始状态线性层
        self.init_state = nn.Linear(image_code_dim, num_layers*hidden_size)
        #2048,512
        print("init_state",image_code_dim,num_layers*hidden_size)

        # GRU层        
        self.rnn = nn.GRU(word_dim + image_code_dim, hidden_size, num_layers)
        print("rnn",word_dim+ image_code_dim,hidden_size,num_layers)
        #self.lstm = nn.LSTMCell(word_dim + image_code_dim, hidden_size, num_layers)

        # Dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 全连接层
        self.fc = nn.Linear(hidden_size, vocab_size)
        print("fc",hidden_size,vocab_size)

        # RNN默认已初始化
        self.init_weights()
        
    def init_weights(self):
        # 初始化权重
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, image_code, captions, cap_lens):
        """
        初始化隐状态
        参数：
            image_code：图像编码器输出的图像表示 
                        (batch_size, image_code_dim, grid_height, grid_width)
        """
        # 将图像表示转换为序列表示形式 
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        # -> (batch_size, image_code_dim) 
        #image_code = image_code.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)
        # -> (batch_size, 1, image_code_dim)
        #image_code = image_code.view(batch_size, -1, image_code_dim)
        # 按照caption的长短排序
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
        # 初始化隐状态
        #print("image_code",image_code.shape)
        #print("image_code.mean(axis=1)",image_code.mean(axis=1).shape)
        #print(image_code_dim)
        hidden_state = self.init_state(image_code)#.mean(axis=1))
        # 重塑隐状态形状
        #print("hidden_state",hidden_state.shape)
        hidden_state = hidden_state.view(
                            batch_size, 
                            self.rnn.num_layers, 
                            self.rnn.hidden_size).permute(1, 0, 2)
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state

    def forward_step(self, image_code, curr_cap_embed, hidden_state):
        # 解码器的前向步骤：以当前时刻词表示为输入，获得GRU输出
        # 将图像编码和当前词的嵌入连接，并增加一个维度
        x = torch.cat((image_code, curr_cap_embed), dim=-1).unsqueeze(0)
        #print(image_code.shape)
        #(batch_size, 1, image_code_dim)
        #print(curr_cap_embed.shape)
        #x = curr_cap_embed.unsqueeze(0)
        # x: (1, real_batch_size, hidden_size+word_dim)
        # out: (1, real_batch_size, hidden_size)
        # 通过GRU层获得输出和更新的隐状态
        out, hidden_state = self.rnn(x, hidden_state)
        # 获取该时刻的预测结果
        # (real_batch_size, vocab_size)
        preds = self.fc(self.dropout(out.squeeze(0)))
        return preds, hidden_state
        
    def forward(self, image_code, captions, cap_lens):
        """
        解码器的前向传播
        参数：
            hidden_state: (num_layers, batch_size, hidden_size)
            image_code:  (batch_size, feature_channel, feature_size)
            captions: (batch_size, )
        """
        # 将图文数据按照文本的实际长度从长到短排序
        # 获得GRU的初始隐状态
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state \
            = self.init_hidden_state(image_code, captions, cap_lens)
        batch_size = image_code.size(0)
        # 输入序列长度减1，因为最后一个时刻不需要预测下一个词
        lengths = sorted_cap_lens.cpu().numpy() - 1
        # 初始化变量：模型的预测结果和注意力分数
        predictions = torch.zeros(batch_size, lengths[0], self.fc.out_features).to(captions.device)
        # 获取文本嵌入表示 cap_embeds: (batch_size, num_steps, word_dim)
        cap_embeds = self.embed(captions)
        # Teacher-Forcing模式有监督
        for step in range(lengths[0]):
            # 解码
            # 模拟pack_padded_sequence函数的原理，获取该时刻的非<pad>输入
            real_batch_size = np.where(lengths>step)[0].shape[0]
            preds, hidden_state = self.forward_step(
                            image_code[:real_batch_size], 
                            cap_embeds[:real_batch_size, step, :],
                            hidden_state[:, :real_batch_size, :].contiguous())            
            # 记录结果
            predictions[:real_batch_size, step, :] = preds
        return predictions, captions, lengths, sorted_cap_indices

# 定义整个模型类
class MYMODL(nn.Module):
    def __init__(self, image_code_dim, vocab, word_dim, hidden_size, num_layers):
        super(MYMODL, self).__init__()
        self.vocab = vocab
        self.encoder = ImageEncoder(models.vgg19(pretrained=True))
        self.decoder = Decoder(image_code_dim, len(vocab), word_dim, hidden_size, num_layers)

    # 前向传播
    def forward(self, images, captions, cap_lens):
        image_code = self.encoder(images)
        return self.decoder(image_code, captions, cap_lens)

    # 使用束搜索生成描述文本
    #images图片输入, beam_k束搜索宽度, max_len句子最大长度
    def generate_by_beamsearch(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.repeat(beam_k,1,1)#.unsqueeze(0) ,1)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            cur_sent_embed = self.decoder.embed(cur_sents)[:,0,:]
            sent_lens = torch.LongTensor([1]*beam_k).to(device)
            # 获得GRU的初始隐状态
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                #print(image_code[:k].squeeze(1).shape)
                #print(cur_sent_embed.shape)
                # 预测下一个词的概率
                preds, hidden_state = self.decoder.forward_step(image_code[:k].squeeze(1), cur_sent_embed, hidden_state.contiguous())
                # -> (k, vocab_size)
                preds = nn.functional.log_softmax(preds, dim=1)
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, vocab_size)
                probs = probs.repeat(1,preds.size(1)) + preds
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    values, indices = probs[0].topk(k, 0, True, True)
                else:
                    # probs: (k, vocab_size) 是二维张量
                    # topk函数直接应用于二维张量会按照指定维度取最大值，这里需要在全局取最大值
                    # 因此，将probs转换为一维张量，再使用topk函数获取最大的k个值
                    values, indices = probs.view(-1).topk(k, 0, True, True)
                # 计算最大的k个值对应的句子索引和词索引
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc') 
                word_indices = indices % vocab_size 
                # 将词拼接在前一轮的句子后，获得此轮的句子
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)
                # 查找此轮生成句子结束符<end>的句子
                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= len(end_indices)
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices) 
                               if word != self.vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    # 仅保留还需要继续生成的句子、句子概率、隐状态、词嵌入
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1,1)
                    hidden_state = hidden_state[:,cur_sent_indices,:]
                    cur_sent_embed = self.decoder.embed(
                        cur_word_indices.view(-1,1))[:,0,:]
                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break
            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_sent = cur_sents[0].tolist()
            else: 
                # 否则选取包含结束符的句子中概率最大的句子
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
        return texts
