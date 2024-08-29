# 特征提取

class ChiSquare:
    def __init__(self, doc_list, doc_labels):
        '''
        初始化ChiSquare类：（基于卡方检验进行特征选择）
        统计所有文档、正类文档和负类文档中各词语出现的频次。
        计算每个词语的卡方统计量，用于后续特征选择。
        
        param: doc_list: 文档列表，每个文档是一个词语列表。
        param: doc_labels: 每个文档对应的标签列表，通常1代表正类，0代表负类。
        '''
        self.total_data, self.total_pos_data, self.total_neg_data = {}, {}, {}

        for i, doc in enumerate(doc_list):
            if doc_labels[i] == 1:
                for word in doc:
                    self.total_pos_data[word] = self.total_pos_data.get(word, 0) + 1
                    self.total_data[word] = self.total_data.get(word, 0) + 1
            else:
                for word in doc:
                    self.total_neg_data[word] = self.total_neg_data.get(word, 0) + 1
                    self.total_data[word] = self.total_data.get(word, 0) + 1

        total_freq = sum(self.total_data.values())
        total_pos_freq = sum(self.total_pos_data.values())
        # total_neg_freq = sum(self.total_neg_data.values())

        self.words = {}

        for word, freq in self.total_data.items():
            pos_score = self.__calculate(self.total_pos_data.get(word, 0), freq, total_pos_freq, total_freq)
            # neg_score = self.__calculate(self.total_neg_data.get(word, 0), freq, total_neg_freq, total_freq)
            self.words[word] = pos_score * 2

    @staticmethod
    def __calculate(n_ii, n_ix, n_xi, n_xx):
        '''
        静态方法，计算给定频次的卡方统计量。
        
        param: n_ii: 在正类中且属于特定词的频次。
        param: n_ix: 在正类中但不属于特定词的频次。
        param: n_xi: 不在正类中但属于特定词的频次。
        param: n_xx: 不在正类中也不属于特定词的频次。
        
        return: chi_square: 卡方统计量。
        '''
        n_ii = n_ii
        n_io = n_xi - n_ii
        n_oi = n_ix - n_ii
        n_oo = n_xx - n_ii - n_oi - n_io
        return n_xx * (float((n_ii*n_oo - n_io*n_oi)**2) /
                       ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo)))

    def best_words(self, num, need_score=False):
        '''
        选取卡方统计量最高的num个词语作为特征。
        
        param: num: 需要选择的特征词数量。
        param: need_score: 是否同时返回词语及其卡方得分，默认False只返回词语。
        
        return: selected_words: 选中的词语列表。如果need_score为True，则同时返回每个词语的卡方得分。
        '''
        words = sorted(self.words.items(), key=lambda word_pair: word_pair[1], reverse=True)

        if need_score:
            return [word for word in words[:num]]
        else:
            return [word[0] for word in words[:num]]







