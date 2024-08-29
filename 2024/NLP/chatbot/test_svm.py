#### test_svm ####
#### pos-precision pos-recall pos-f1 ####
#### neg-precision neg-recall neg-f1 ####
#### total-recall ####

import jieba
import pandas as pd

from spa.classifiers import SVMClassifier
svm = SVMClassifier()

#### 单个预测 ####
def get_predict_label(user_input):
    score, classify_label = svm.classify(user_input)
    #print("Score", score, ", 极性：", classify_label)
    return classify_label

def load_and_process_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)

    user_inputs = data.iloc[:, 1].values 
    origin_labels = data.iloc[:, 0].values 

    classify_labels = []
    for user_input in user_inputs:
        # 使用jieba分词
        words = [word for word in jieba.lcut(user_input)]
        classify_labels.append(get_predict_label(words))
        
    return origin_labels, classify_labels

#### 评测指标 ####
def get_accuracy(origin_labels, classify_labels):
    assert len(origin_labels) == len(classify_labels)

    xls_contents = []

    pos_right, pos_false = 0, 0
    neg_right, neg_false = 0, 0
    for i in range(len(origin_labels)):
        if origin_labels[i] == 1:
            if classify_labels[i] == 1:
                pos_right += 1
            else:
                neg_false += 1
        else:
            if classify_labels[i] == 0:
                neg_right += 1
            else:
                pos_false += 1
    xls_contents.extend([("neg-right", neg_right), ("neg-false", neg_false)])
    xls_contents.extend([("pos-right", pos_right), ("pos-false", pos_false)])

    pos_precision = pos_right / (pos_right + pos_false) * 100
    pos_recall = pos_right / (pos_right + neg_false) * 100
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
    xls_contents.extend([("pos-precision", pos_precision), ("pos-recall", pos_recall), ("pos-f1", pos_f1)])

    neg_precision = neg_right / (neg_right + neg_false) * 100
    neg_recall = neg_right / (neg_right + pos_false) * 100
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
    xls_contents.extend([("neg-precision", neg_precision), ("neg-recall", neg_recall), ("neg-f1", neg_f1)])

    total_recall = (neg_right + pos_right) / (neg_right + neg_false + pos_right + pos_false) * 100
    xls_contents.append(("total-recall", total_recall))

    print("    pos-right\tpos-false\tneg-right\tneg-false\tpos-precision\tpos-recall\t"
          "pos-f1\tneg-precision\tneg-recall\tneg-f1\ttotal-recall")
    print("    " + "---" * 45)
    print("    %8d\t%8d\t%8d\t%8d\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f\t%8.4f" %
          (pos_right, pos_false, neg_right, neg_false, pos_precision, pos_recall,
           pos_f1, neg_precision, neg_recall, neg_f1, total_recall))

    return xls_contents

if __name__ == '__main__':
    test_file = ['spa/waimai_10k.csv', 'spa/AmbiguousCustomerSentences_Reformatted.csv']

    print(f'Test on {test_file[0]}')
    origin_labels0, classify_labels0 = load_and_process_data(test_file[0])
    x0 = get_accuracy(origin_labels0, classify_labels0)
    print(x0)

    print(f'Test on {test_file[1]}')
    origin_labels1, classify_labels1 = load_and_process_data(test_file[1])        
    x1 = get_accuracy(origin_labels1, classify_labels1)
    print(x1)
    
