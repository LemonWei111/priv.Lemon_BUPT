from nltk import ngrams
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu

def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]

############################################## bleu4 ################################################3

def evaluate_bleu_4(dataset, completions_per_prompt=1):

    # 存储候选文本
    cands = []
    # 存储参考文本
    refs = []
    # 需要过滤的词
    filterd_words = set({'\n'})
    cpi = completions_per_prompt

    for data in dataset:
        completion = data['completion']
        hypothesis = data['hypothesis']
        # 候选文本
        cands.extend([filter_useless_words(hypothesis, filterd_words)])
        # 参考文本
        refs.extend([filter_useless_words(completion, filterd_words)])

    # 实际上，每个候选文本对应cpi条参考文本
    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])

    # 计算BLEU-4值，corpus_bleu函数默认weights权重为(0.25,0.25,0.25,0.25)
    # 即计算1-gram到4-gram的BLEU几何平均值
    bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))

    return bleu4

######################################################    ROUGE-L    ###################################################

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

    def calc_precision_recall_f1(reference_list, hypothesis_list, n=1):
        """计算Precision, Recall, F1 for ROUGE-n"""
        ref_counts = Counter(get_ngrams(reference_list, n))
        hyp_counts = Counter(get_ngrams(hypothesis_list, n))
   
        overlap = sum((ref_counts & hyp_counts).values())
   
        prec = overlap / float(len(hyp_counts)) if hyp_counts else 0.0
        rec = overlap / float(len(ref_counts)) if ref_counts else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if prec + rec else 0.0
   
        return prec, rec, f1

    """计算ROUGE-1, ROUGE-2"""
    rouge_1 = calc_precision_recall_f1(reference, candidate, 1)
    rouge_2 = calc_precision_recall_f1(reference, candidate, 2)

    return rouge_1[2], rouge_2[2]

#默认cpi = 1
def evaluate_rouge_l(dataset):

    # 需要过滤的词
    filterd_words = set({'\n'})

    rouge_1_scores = []
    rouge_2_scores = []

    for data in dataset:
        completion = data['completion']
        hypothesis = data['hypothesis']
        
        # 候选文本
        cands = filter_useless_words(hypothesis, filterd_words)
        # 参考文本
        refs = filter_useless_words(completion, filterd_words)

        #print(refs, cands)

        rouge_1, rouge_2 = calculate_rouge_l(refs, cands)

        rouge_1_scores.append(rouge_1)
        rouge_2_scores.append(rouge_2)

    # 计算平均ROUGE-L得分
    average_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if len(rouge_1_scores) > 0 else 0
    average_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores) if len(rouge_2_scores) > 0 else 0

    return average_rouge_1, average_rouge_2

#外部调用：
#from metric import evaluate_bleu_4, evaluate_rouge_l, fromtexttosentence

#evaluate_rouge_l默认对图片只有一条参考描述，适用于本例内容

#参数情况：
#evaluate_bleu_4(dataset, completions_per_prompt)
#evaluate_rouge_l(dataset)
