import os
import pickle
import numpy as np

from spa.feature_extraction import ChiSquare

# ################################################
# 基于SVM的情感极性分析
# ################################################
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, train_data=None, train_labels=None, feature_num=5000, C=50):
        '''
        初始化SVM分类器：
        检查预训练模型是否存在，不存在则进行模型训练并保存；存在则直接加载模型。
        使用定义的ChiSquare进行特征选择。
        
        param: train_data: 训练数据集，每条数据为一个文本样本的词语列表。
        param: train_labels: 训练数据集对应的标签列表（0/1）。
        param: feature_num: 选择的特征数量，默认为5000。
        param: C: SVM的正则化参数，默认为50。
        '''
        # 当前执行目录
        rootdir = os.getcwd()
        model_path = os.path.join(rootdir, "spa/svm_model.pkl")
        print(model_path)

        if not os.path.exists(model_path):
            # feature extraction
            fe = ChiSquare(train_data, train_labels)
            self.best_words = fe.best_words(feature_num)

            train_data = np.array(train_data)
            train_labels = np.array(train_labels)
            
            self.C=C
            
            self.clf = SVC(C=self.C, probability=True)
            self.__train(train_data, train_labels)
            self.__save_model(model_path)
        else:
            self.__load_model(model_path)
            
    def words2vector(self, all_data):
        '''
        将文本数据转化为向量形式。
        
        param: all_data: 文本数据列表，每个元素为一个文本样本的词语列表。
        
        return: vectors: 转换后的向量列表，每个向量对应一个样本。
        '''
        vectors = []

        best_words_index = {}
        for i, word in enumerate(self.best_words):
            best_words_index[word] = i

        for data in all_data:
            vector = [0 for x in range(len(self.best_words))]
            for word in data:
                i = best_words_index.get(word)
                if i is not None:
                    vector[i] = vector[i] + 1
            vectors.append(vector)

        vectors = np.array(vectors)
        return vectors

    def __train(self, train_data, train_labels):
        '''
        训练SVM模型。
        
        param: train_data: 训练数据的向量化表示。
        param: train_labels: 训练数据的标签列表。
        '''
        print("SVMClassifier is training ...... ")

        train_vectors = self.words2vector(train_data)

        self.clf.fit(train_vectors, np.array(train_labels))

        print("SVMClassifier trains over!")

    def classify(self, data):
        '''
        对单个样本进行分类预测。
        
        param: data: 单个文本样本的词语列表。
        
        return: proba: 预测为正类的概率。
        return: prediction: 预测的类别标签。
        '''
        vector = self.words2vector([data])

        prediction = self.clf.predict(vector)
        # print(self.clf.predict_proba(vector))

        return self.clf.predict_proba(vector)[0][1], prediction[0]

    def __save_model(self, file_path):
        '''
        保存模型至文件。
        
        param: file_path: 模型保存的文件路径。
        '''
        model_data = {
            'best_words': self.best_words,
            'C': self.C
        }
        with open(file_path, 'wb') as f:
            pickle.dump((model_data, self.clf), f)

    def __load_model(self, file_path):
        '''
        从文件加载模型。
        
        param: file_path: 模型文件的路径。
        '''
        with open(file_path, 'rb') as f:
            model_data, clf = pickle.load(f)
        self.best_words = model_data['best_words']
        self.C = model_data['C']
        self.clf = clf
