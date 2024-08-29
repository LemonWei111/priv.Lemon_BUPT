# BERT+CRF 模型定义
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from TorchCRF import CRF

# 定义CRF层 TODO
class my_CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones(emissions.size(0), emissions.size(1)) # , dtype=torch.bool)
        score = self._score(emissions, tags, mask)
        forward_score = self._forward_score(emissions, mask)
        loss = torch.mean(forward_score - score)
        return loss

    def _score(self, emissions, tags, mask):
        # emissions: (batch_size, seq_len, num_tags)
        # tags: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        # print(tags.size())
        batch_size, seq_len = tags.size()
        score = torch.zeros(batch_size)
        
        # print(score.size(), self.start_transitions[tags[:,0]].size(), self.start_transitions[tags[:,0]])
        score += self.start_transitions[tags[:, 0]]
        
        # 对发射分数进行log-softmax，以便进行后续的分数计算
        emissions = emissions.log_softmax(dim=2)
        
        # print(emissions.shape, emissions)
        # print(torch.arange(batch_size).shape, torch.arange(seq_len).shape, tags.shape)
        # print('e', emissions[torch.arange(batch_size)[:,None], torch.arange(seq_len)[None:], tags].shape)
        # print('m', mask.shape)
        # print('s', score.shape)
        
        emissions_scores = emissions[torch.arange(batch_size)[:, None], torch.arange(seq_len)[None, :], tags] * mask

        score += torch.sum(emissions_scores, dim=1)
        for i in range(1, seq_len):
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
        score += self.end_transitions[tags[:, -1]] * mask[:, -1]
        return score
    
    def decode(self, emissions, mask=None):
        # emissions: (batch_size, seq_len, num_tags)
        # mask: (batch_size, seq_len)
        if mask is None:
            mask = torch.ones(emissions.size(0), emissions.size(1)) # , dtype=torch.bool)

        # 初始化维特比变量
        batch_size, seq_len, num_tags = emissions.size()
        vit = torch.zeros((batch_size, num_tags))
        # print(vit.shape, self.start_transitions.shape, emissions[:, 0].shape)
        vit = self.start_transitions + emissions[:, 0]
   
        # 为每个时间步计算维特比变量
        backpointers = []
        for i in range(1, seq_len):
            vit_exp = vit.unsqueeze(2) + self.transitions
            vit_exp, backpointer = vit_exp.max(dim=1)
            backpointers.append(backpointer)
            # print(mask[:, i])
            vit = vit_exp * mask[:, i].unsqueeze(1) + vit * (1-(mask[:, i])).unsqueeze(1)
            vit += emissions[:, i]

        # 加上结束状态的转换分数
        vit += self.end_transitions

        # 找出最可能的标签序列
        paths = []
        for i in range(batch_size):
            path = [vit[i, :].argmax().item()]
            for bp in reversed(backpointers):
                path.append(bp[i, path[-1]].item())
            paths.append(path[::-1])
        return paths

    def _forward_score(self, emissions, mask):
        # emissions: (batch_size, seq_len, num_tags)
        # mask: (batch_size, seq_len)
        batch_size, seq_len, num_tags = emissions.size()
        alpha = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_len):
            alpha = (alpha.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)).logsumexp(dim=1)
            alpha_masked = alpha * mask[:, i].unsqueeze(1)
        alpha_lse = (alpha_masked + self.end_transitions).logsumexp(dim=1)        
            # alpha *= mask[:, i].unsqueeze(1)
        # alpha = (alpha + self.end_transitions).logsumexp(dim=1)
        return alpha_lse

        
# 定义BERT配置
config = BertConfig(
    vocab_size=21128, # BERT-base-chinese的词汇表大小
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    from_scratch=True # 设置为True以随机初始化模型
)

# 定义BERT_NER模型
class BERT_NER(nn.Module):
    def __init__(self, num_tags, drop=0.1):
        '''
        定义BERT_NER模型，用于命名实体识别任务。
        
        :param num_tags: int 输出标签的数量
        :param drop: float, optional 丢弃率，默认为0.1
        '''
        super(BERT_NER, self).__init__()
        
        # 不使用预训练模型
        # self.bert = BertModel(config)
        # 使用预训练的中文BERT模型
        self.bert = BertModel.from_pretrained('bert-base-chinese') # ./huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f') # ('bert-base-chinese', force_download=True, resume_download=False)

        # 设置Dropout层，用于防止过拟合
        # 线性层用于将BERT输出映射到标签空间
        # CRF层，用于NER标签的推断
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags) # , batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        '''
        前向传播过程
        
        :param input_ids: tensor 输入的词/字的索引
        :param attention_mask: tensor 输入的注意力掩码
        :param labels: tensor, optional 输入的标签，用于计算损失，默认为None（预测）

        :return: 损失值（训练）/预测的最佳路径（预测）
        '''
        # 将输入传入BERT模型并获取最后一层的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]  # 只取最后一层的输出
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        # print(logits)

        if labels is not None:
            # 计算CRF损失
            loss = -self.crf(logits, labels, attention_mask.bool())
            return loss
        else:
            # 使用Viterbi算法获取最佳路径（用于预测阶段）
            best_paths = self.crf.viterbi_decode(logits, attention_mask.bool())
            return best_paths

def correct_num(predicted_labels, labels):
    '''
    计算每个序列中预测标签与实际标签匹配的数量（本实验只在最后一行PAD，因而在计算correct_num时近似忽略mask）

    :param predicted_labels: list 预测标签列表
    :param labels: list 真实标签列表

    :return 预测正确数
    '''
    # 初始化 correct_predictions 为 0
    correct_predictions = 0
    # 遍历每个序列
    for predicted_label, label in zip(predicted_labels, labels):
        # print(predicted_label, label)
        # 初始化匹配计数为 0
        match_count = 0
        # 遍历序列中的每个标签
        for p, l in zip(predicted_label, label):
            # print(p, l)
            # 如果预测标签和实际标签相等，则增加匹配计数
            if p == l:
                match_count += 1
        # 将当前序列的匹配计数累加到 correct_predictions 中
        correct_predictions += match_count

    # print(correct_predictions)
    return correct_predictions

'''
#use exmaple

# 初始化模型
num_tags = 4  # 假设有9个不同的标签
model = BERT_NER(num_tags)

# 为了演示，我们创建一些假的输入数据
# 注意：在实际使用中，您需要根据实际数据来创建这些输入
input_ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 4]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
input_labels = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 3]])
# labels = nn.functional.one_hot(input_labels, num_classes = num_tags)
# 前向传播
# print(model(input_ids, attention_mask, input_labels))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

###################    train&test    #################
# 训练模型
num_epochs = 100 # 训练的轮数
torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    model.train() # 设置模型为训练模式

    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask, labels=input_labels)
    out_loss = outputs.loss if isinstance(outputs, dict) else outputs
    loss = torch.mean(out_loss)
    # print(loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    loss.backward()
    optimizer.step()

    if epoch%10==0:
        model.eval() # 设置模型为评估模式
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
        
            outputs = model(input_ids, attention_mask)
            predicted_labels = outputs.logits if isinstance(outputs, dict) else outputs

            # print(predicted_labels, input_labels, predicted_labels==input_labels)
            correct_predictions += correct_num(predicted_labels, input_labels)
            # correct_predictions += (predicted_labels == labels).sum(dim=1).sum().item()
            total_predictions += input_labels.size(1) * input_labels.size(0)
            accuracy = correct_predictions / total_predictions
            print(f"Dev Accuracy: {accuracy:.4f}")
'''
