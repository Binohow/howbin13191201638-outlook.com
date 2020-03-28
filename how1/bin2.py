""" 英文文本数据处理 """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.preprocessing.text as text
import numpy as np
from keras.utils import to_categorical
txt = "Those that come through the Ivory Gate cheat us with empty promises \
that never see fulfillment. Those that come through the Gate of Horn inform \
the dreamer of the truth"
# %% 分词
out1 = text.text_to_word_sequence(txt)
# %% 建立索引
myset = set(out1)  # 利用set去重复
out2 = list(myset) 
out2.sort()      
mydict1 = dict(list(zip(np.arange(len(out1)),out2)))
mydict2 = dict(list(zip(out2,np.arange(len(out1)))))
"""
首先通过 zip 命令将每个单词依次与序号配对，然后通过 list 命令将配对的数据改为列表，每
个元素是诸如 ('with', 0) 这样的一对，最后应用字典命令将列表修改为字典即可完成
索引。
"""
# %% ont hot 编码
word_num=list()
for word in out1:
    word_num.append(mydict2.get(str(word)))
# data 为txt的序列    
data = np.array(word_num)
#编码
encoded = to_categorical(data)
#解码
inverted = np.argmax(encoded[0])
