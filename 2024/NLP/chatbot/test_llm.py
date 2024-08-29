#### test_llm ####
#### Rouge-1 Rouge-2 Bleu-4 ####

import time
import jieba
import pandas as pd

from api import call_with_messages
from metric import evaluate_bleu_4, evaluate_rouge_l

def load_and_process_data(file_path, model_name='llama3-8b-instruct', use_prompt=True, use_score=True):
    if not use_prompt:
        use_score = False
        
    if use_score:
        from spa.classifiers import SVMClassifier
        svm = SVMClassifier()
        
    # 从Excel文件加载数据
    df = pd.read_excel(file_path)

    # 初始化一个列表来存储处理后的数据
    dataset = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        user_input = row['Prompt'] # 假设第一列名为'Prompt'
        completion = row['Completion'] # 假设第二列名为'Completion'
        
        if use_score:
            # score, _ = svm.classify(user_input) # 逐字输入SVM
            score, _ = svm.classify([word for word in jieba.lcut(user_input)]) # 按jieba分词输入SVM
            user_input = f'{score}'+' '+user_input

        # 调用call_with_messages函数生成'hypothesis'
        hypothesis = call_with_messages(user_input, model_name, use_prompt)
        #print(hypothesis)

        # 将处理后的数据添加到dataset列表中
        dataset.append({'completion': completion, 'hypothesis': hypothesis})

        # 防止频繁调用API引起的调用失败
        time.sleep(5)
        
    return dataset

def calculate_average_metrics(dataset):
    """
    计算多组参考文本与生成文本的平均评价指标。

    :param dataset: 包含多组{'completion': str, 'hypothesis': str}的列表
    :return: 所有数据组的平均评价指标
    """
    # 计算平均分
    avg_rouge1, avg_rouge2 = evaluate_rouge_l(dataset)
    avg_bleu4 = evaluate_bleu_4(dataset)

    return {"Average Rouge-1": avg_rouge1, "Average Rouge-2": avg_rouge2, "Average Bleu-4": avg_bleu4}

if __name__ == '__main__':
    # 使用函数处理Excel文件
    file_path = 'Testdata.xlsx'
    choose_model = ['qwen1.5-1.8b-chat', 'llama3-8b-instruct']
    
    d1 = load_and_process_data(file_path, choose_model[1], False, False)
    r1 = calculate_average_metrics(d1)
    print(f"原始{choose_model[1]}：", r1)
    
    d2 = load_and_process_data(file_path, choose_model[0], False, False)
    r2 = calculate_average_metrics(d2)
    print(f"原始{choose_model[0]}：", r2)

    d3 = load_and_process_data(file_path, choose_model[1], True, False)
    r3 = calculate_average_metrics(d3)
    print(f"指令{choose_model[1]}：", r3)

    d4 = load_and_process_data(file_path, choose_model[0], True, False)
    r4 = calculate_average_metrics(d4)
    print(f"指令{choose_model[0]}：", r4)
    
    d5 = load_and_process_data(file_path, choose_model[1], True, True)
    r5 = calculate_average_metrics(d5)
    print(f"情感{choose_model[1]}：", r5)

    d6 = load_and_process_data(file_path, choose_model[0], True, True)
    r6 = calculate_average_metrics(d6)
    print(f"情感{choose_model[0]}：", r6)
    
    
