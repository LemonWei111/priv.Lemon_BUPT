# Text Evaluation Utilities
# Version: 1.1
# Author: [魏靖]
#Change：1.0-1.1添加get_keys_by_value、fromtexttosentence、fromtexttosentencetext、calculate_rouge_l

# 导入必要的 PyTorch 模块和第三方库
import torch

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk import ngrams

# 过滤句子中的无效词汇
def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU等评测值计算的符号
    return [w for w in sent if w not in filterd_words]

############################################## bleu4 ################################################
def evaluate_bleu_4(data_loader, model, captions_per_image, beam_k, max_len):
    model.eval()
    # 存储候选文本
    cands = []
    # 存储参考文本
    refs = []
    # 需要过滤的词
    filterd_words = set({model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']})
    cpi = captions_per_image
    device = next(model.parameters()).device
    for i, (imgs, caps, caplens) in enumerate(data_loader):
        with torch.no_grad():
            # 通过束搜索，生成候选文本
            texts = model.generate_by_beamsearch(imgs.to(device), beam_k, max_len+2)
            # 候选文本
            cands.extend([filter_useless_words(text, filterd_words) for text in texts])
            # 参考文本
            refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])
    # 实际上，每个候选文本对应cpi条参考文本
    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
    # 计算BLEU-4值，corpus_bleu函数默认weights权重为(0.25,0.25,0.25,0.25)
    # 即计算1-gram到4-gram的BLEU几何平均值
    bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
    model.train()
    return bleu4

# 通过值获取字典中的键
def get_keys_by_value(dict_obj, value):
    for k, v in dict_obj.items():
        if v == value:
            return k

# 将整数序列转换为文本句子
#输出字符串
def fromtexttosentence(text, words):
    sentence = ""
    for i in text[0]:
        #print(get_keys_by_value(words, i), end=" ")
        sentence += get_keys_by_value(words, i) + " "
    return sentence

# 将整数序列转换为文本字符列表
#输出字符列表
def fromtexttosentencetext(text, words):
    sentence = []
    #print(type(sentence))
    for i in text[0]:
        #print(type(sentence))
        #赋值一个变量，再用点操作符调用原对象的方法，这个时候会报错 'NoneType'
        #print(get_keys_by_value(words, i), end=" ")
        sentence.append(get_keys_by_value(words, i))
        #print(sentence)
    #del sentence[0]
    return sentence
#use example
#text = [[111, 42, 34, 14, 41, 34, 110, 23, 112]]
#fromtexttosentence(text, all_words)
#fromtexttosentencetext(text, all_words)

######################################################    ROUGE-L    ###################################################
# 计算 ROUGE-L 得分
#输入string
def calculate_rouge_l(reference, candidate):
    """
    计算ROUGE-L评测指标

    参数:
    - reference: 参考文本，形如[['word1', 'word2', ...], ...]
    - candidate: 候选文本，形如['word1', 'word2', ...]

    返回:
    - rouge_l: ROUGE-L得分
    """

    def get_ngrams(tokens, n):
        """
        获取n-grams
        """
        ngrams_list = ngrams(tokens, n)
        return [' '.join(gram) for gram in ngrams_list]

    def compute_lcs(reference, candidate):
        """
        计算最长公共子序列（LCS）
        """
        reference_ngrams = set(get_ngrams(reference, 1))
        candidate_ngrams = set(get_ngrams(candidate, 1))
        lcs = reference_ngrams.intersection(candidate_ngrams)
        return len(lcs)

    total_lcs = 0
    total_reference_length = 0

    for ref_set in reference:
        lcs = compute_lcs(ref_set, candidate)
        total_lcs += lcs
        total_reference_length += len(ref_set)

    precision = total_lcs / len(candidate) if len(candidate) > 0 else 0
    recall = total_lcs / total_reference_length if total_reference_length > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

# ROUGE-L 评测函数
#默认cpi = 1
def evaluate_rouge_l(data_loader, model, beam_k, max_len):
    model.eval()
    rouge_l_scores = []

    device = next(model.parameters()).device
    # 需要过滤的词
    filterd_words = set({model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']})

    for i, (imgs, caps, caplens) in enumerate(data_loader):
        with torch.no_grad():
            # 通过束搜索，生成候选文本
            texts = model.generate_by_beamsearch(imgs.to(device), beam_k, max_len + 2)
            # 候选文本的整数序列
            cands = [filter_useless_words(text, filterd_words) for text in texts]
            # 参考文本的整数序列
            refs = [filter_useless_words(cap, filterd_words) for cap in caps.tolist()]
            #print(cands)
            #print(refs)
            s_cands = fromtexttosentencetext(cands, model.vocab)
            s_refs = fromtexttosentencetext(refs, model.vocab)

            rouge_l = calculate_rouge_l(s_refs, s_cands)
            #print(rouge_l)
            rouge_l_scores.append(rouge_l)

    # 计算平均ROUGE-L得分
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if len(rouge_l_scores) > 0 else 0

    model.train()
    return average_rouge_l

#外部调用：
#from evaluate import evaluate_bleu_4, evaluate_rouge_l, fromtexttosentence

#evaluate_rouge_l默认对图片只有一条参考描述，适用于本例内容
#fromtexttosentence可用于根据整数序列展示输出

#参数情况：
#evaluate_bleu_4(data_loader, model, captions_per_image, beam_k, max_len)
#evaluate_rouge_l(data_loader, model, beam_k, max_len)
#fromtexttosentence使用示例参见原函数定义之后
