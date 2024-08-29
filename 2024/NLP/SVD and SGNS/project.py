# word2vec for sgns
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 训练Word2Vec模型
sentences = LineSentence('./training.txt')  # 替换为你的训练数据文件路径
model = Word2Vec(sentences, vector_size=5, window=2, min_count=2, workers=3)

def write_vector2file(vector_path, model):
    with open(vector_path, 'w', encoding='utf-8') as outfile:
        for word in model.wv.index_to_key:
            vector = model.wv[word]
            # print(word, vector)
            word_line = f"{word} "
            outfile.write(word_line)
            for v in vector:
                vector_line = f"{v} "
                outfile.write(vector_line)
            # 写入新文件
            outfile.write("\n")

write_vector2file("./sgns.txt", model)
