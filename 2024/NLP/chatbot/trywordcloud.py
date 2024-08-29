import jieba
import jieba.analyse
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

import chardet
text_path = 'message.txt'

def detect_encoding(file_path):
    '''
    使用chardet库打开文件并检测其二进制内容的编码

    param: file_path  文件路径字符串

    return: 检测到的文件编码，若未检测到则默认为'utf-8'
    '''
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    if encoding is None:
        encoding = 'utf-8'
    return encoding

def get_keywords(text_path, topK = 10):
    '''
    从文本文件中提取关键词并生成词云
    #   1. 检测停用词文件和文本文件的编码
    #   2. 加载停用词列表
    #   3. 读取并分词文本，去除停用词
    #   4. 使用jieba提取关键词及其TF-IDF值
    #   5. 打印关键词及其TF-IDF值，并保存到文件
    #   6. 生成并显示词云图像，可选保存到本地

    param: text_path  待分析文本文件路径
    param: topK  需要提取的关键词数量，默认为10
    '''
    stopwords_path = 'stopwords.txt'

    stopwords_encoding = detect_encoding(stopwords_path)
    text_encoding = detect_encoding(text_path)
    print(f"The encoding of the file is: {stopwords_encoding}")
    print(f"The encoding of the file is: {text_encoding}")

    # 加载停用词列表
    with open(stopwords_path, 'r', encoding=stopwords_encoding) as f:
        stopwords = set([line.strip() for line in f.readlines()])
        print(stopwords)

    # 打开文本：指定使用utf-8编码读取
    with open(text_path, 'r', encoding=text_encoding) as f:
        text = f.read()
    
    # 使用jieba分词并去除停用词
    words = [word for word in jieba.lcut(text) if word not in stopwords]

    # 提取关键词
    keywords = jieba.analyse.extract_tags(' '.join(words), topK=topK, withWeight=True)

    # 构建关键词及其TF-IDF值的字典
    keyword_dict = dict(keywords)    

    # 打印关键词及其TF-IDF值
    # 保存关键词及其TF-IDF值到文本文件
    with open('keyword_tfidf_freq.txt', 'w', encoding='utf-8') as file:
        for keyword, score in keyword_dict.items():
            print(f"'{keyword}': {score}")
            file.write(f"{keyword} {score}\n")

    # 准备词云文本，仅使用关键词
    text_for_wordcloud = ' '.join(keyword for keyword in keyword_dict.keys())

    # 生成词云
    font_path='MaShanZheng-Regular.ttf'
    wc = WordCloud(font_path=font_path, width=800, height=600, background_color='white')
    wc.generate(text_for_wordcloud)

    # 显示词云
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # 可选：保存词云图像到文件
    wc.to_file('keyword_wordcloud.png')

def generate_wordcloud_from_keywords(file_path='keyword_list.json', image_path='wordcloud.png', debug=False):
    '''
    从关键词列表json文件生成词云并保存为图片。

    :param file_path: 关键词列表的json文件路径。
    :param image_path: 生成的词云图片保存路径。
    '''
    # 从json文件加载关键词和对应的权重
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords_data = json.load(file)
       
    # 准备词云文本，格式为单词:权重
    words_for_cloud = {word: weight for word, weight in keywords_data.items()}
    print(words_for_cloud)
    
    font_path='MaShanZheng-Regular.ttf'

    # 创建WordCloud实例
    wc = WordCloud(
        background_color='white', # 设置背景颜色
        width=800, height=600, # 设置图片宽度和高度
        font_path=font_path, # 设置字体路径，确保可以显示中文
        max_words=200, # 词云中最多显示的词数
        random_state=42, # 设置随机种子以便复现实验结果
    )
   
    # 生成词云
    wordcloud = wc.generate_from_frequencies(words_for_cloud)

    if debug:
        # 显示词云（可选，通常在调试时使用）
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
   
    # 保存词云到图片文件
    wordcloud.to_file(image_path)
    print(f"词云已成功保存至：{image_path}")

def load_keywords(file_path='keyword_list.json'):
    '''从文件加载关键词列表'''
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_keywords(keywords, file_path='keyword_list.json', maxnum=50):
    '''保存关键词列表到文件，按TF-IDF值降序排列并限制数量至maxnum'''
    # 确保关键词列表按TF-IDF值降序排列
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    # 限制关键词数量
    limited_keywords = {word: score for word, score in sorted_keywords[:maxnum]}
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(limited_keywords, file, ensure_ascii=False, indent=4)

def update_keywords_with_tfidf(text, existing_keywords, topK=10, maxnum=50):
    '''
    更新关键词列表，包括新提取的关键词以及历史关键词在当前文本中的TF-IDF值，
    并确保最终关键词数量不超过maxnum。
    '''
    # 提取关键词
    new_keywords = jieba.analyse.extract_tags(' '.join(text), topK=topK, withWeight=True)
    new_keywords_dict = {word: score for word, score in new_keywords}

   
    # 维护一个包含新旧关键词的集合，用于构建计算TF-IDF的文本
    all_keywords_set = set(existing_keywords.keys()).union(set(new_keywords_dict.keys()))
    print(all_keywords_set)
   
    vectorizer = TfidfVectorizer(use_idf=True, vocabulary=all_keywords_set)
    tfidf_matrix = vectorizer.fit_transform(text)
    tfidf_values = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
   
    # 更新关键词列表
    for word in all_keywords_set:
        if word in new_keywords_dict:
            existing_keywords[word] = new_keywords_dict[word]
        elif word in tfidf_values:
            existing_keywords[word] = max(existing_keywords.get(word, 0), tfidf_values[word])
        else:
            existing_keywords[word] = existing_keywords.get(word, 0) + 0.01 # 保守递增策略
   
    # 按TF-IDF值排序并限制数量
    sorted_keywords = sorted(existing_keywords.items(), key=lambda x: x[1], reverse=True)[:maxnum]
    existing_keywords = dict(sorted_keywords)
   
    return existing_keywords

def read_text(text_path):
    '''
    根据文件的实际编码读取文本内容
    尝试使用自动检测的编码读取文件，若失败则尝试使用'utf-8'

    :param text_path  文本文件路径
    
    :return 文件的文本内容
    '''
    text_encoding = detect_encoding(text_path)
    print(text_encoding)
    try:
        text = open(text_path, 'r', encoding=text_encoding).read()
    except:
        text = open(text_path, 'r', encoding='utf-8').read()
    return text

def process_text_and_update_keywords(text_path, keyword_path='keyword_list.json', maxnum=50, debug=False):
    '''
    处理文本，更新并保存关键词，生成词云
    #   1. 检测停用词文件编码
    #   2. 读取文本内容并分词去停用词
    #   3. 更新关键词列表并保存
    #   4. 调用generate_wordcloud_from_keywords生成词云

    :param text_path  待处理的文本文件路径
    :param keyword_path  存储关键词的json文件路径
    :param maxnum  关键词列表的最大数量
    :param debug  是否开启调试模式显示词云
    '''
    stopwords_path = 'stopwords.txt'
    stopwords_encoding = detect_encoding(stopwords_path)
    # 加载停用词列表
    with open(stopwords_path, 'r', encoding=stopwords_encoding) as f:
        stopwords = set([line.strip() for line in f.readlines()])
        print(stopwords)

    text = read_text(text_path)
    # 使用jieba分词并去除停用词
    words = [word for word in jieba.lcut(text) if word not in stopwords]

    if os.path.exists(keyword_path):
        existing_keywords = load_keywords(keyword_path)
    else:
        existing_keywords = {}
        
    updated_keywords = update_keywords_with_tfidf(words, existing_keywords, maxnum=maxnum)
    save_keywords(updated_keywords, maxnum=maxnum)

    # 调用函数生成词云
    generate_wordcloud_from_keywords('keyword_list.json', 'my_wordcloud.png', debug=debug)


if __name__ == '__main__':
    # 调用函数处理文本并更新关键词列表，限制最大关键词数量
    process_text_and_update_keywords(text_path, 'keyword_list.json', maxnum=100, debug=True) # 示例中设置maxnum为100

    # 单次关键词提取与词云生成
    # get_keywords(text_path)
