'''
文本生成实例：爱丽丝梦游仙境
LSTM
'''
from nltk import data
data.path.append(r"Y:\python\NLTK\nltk_data")
from nltk import word_tokenize
from gensim import corpora
from pyecharts.charts import WordCloud
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils

filename='Alice.txt'
dict_file = 'dict_file.txt'
model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'

batch_size = 128
epochs = 200
dict_len = 3281
max_len = 20
document_max_len = 33200


def clear_data(str):
    '''
        删除字符串中的特殊字符或换行符
    '''
    value = str.replace('\ufeff', '').replace('\n', '')
    return value
    
def load_dataset(): 
    '''
        导入数据
    '''
    with open(filename, mode='r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            # 删除非内容字符
            value = clear_data(line)
            if value != '':
                # 对一行文本进行分词
                for str in word_tokenize(value):
                    # 跳过章节标题
                    if str == 'CHAPTER':
                        break
                    else:
                        document.append(str.lower())
    return document

def word_to_integer(document):
    '''
    :return 单词向量
    '''
    dic = corpora.Dictionary([document])
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    for word in document:
    # 查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values

def show_word_cloud(document):
    # 需要清除的标点符号
    left_words = ['.', ',', '?', '!', ';', ':', '\'', '(', ')']
    # 生成字典
    dic = corpora.Dictionary([document])
    # 计算得到每个单词的使用频率
    words_set = dic.doc2bow(document)
    # 生成单词列表和使用频率列表
    my_words=list()
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            my_words.append((word,frequence))
            
    # 使用 pyecharts 生成词云
    word_cloud = WordCloud()
    word_cloud.add(series_name='Alice\'s word cloud', data_pair=my_words, shape='circle', word_size_range=[20, 100], width=1000, height=620)
    word_cloud.render("wordcloud_custom_mask_image.html")

def build_model():
    model = Sequential()
    model.add(
        Embedding(input_dim=dict_len, output_dim=32, input_length=max_len)
    )
    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu'
        )
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=dict_len, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def  make_dataset(document):
    '''
        :return 按照固定长度拆分文本
    '''
    dataset =  np.array(document[0:document_max_len])
    dataset=dataset.reshape(int(document_max_len / max_len), max_len)
    return dataset

def make_y(document):
    dataset = make_dataset(document)
    y = dataset[1:dataset.shape[0], 0]
    return y

def make_x(document):
    dataset = make_dataset(document)
    x = dataset[0: dataset.shape[0] - 1, :]
    return x

document = load_dataset()
values = word_to_integer(document)

x_train = make_x(values)
y_train = make_y(values)
y_train = np_utils.to_categorical(y_train, dict_len)
model = build_model()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=2)
# 保存模型到 json 文件中
model_json = model.to_json()
with open(model_json_file, 'w') as file:
    file.write(model_json)
model_json = model.to_json()
# 保存权重数值到文件中
model.save_weights(model_hd5_file)
