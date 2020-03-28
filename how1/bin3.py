import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.preprocessing.text as text
import numpy as np
from keras.utils import to_categorical
import jieba
from pypinyin import lazy_pinyin

def pinyin_sort(lis):             #输入一个名字的列表
    """ 中文拼音排序 """
    result=[]
    for item in lis:
        result.append((lazy_pinyin(item),item))
    result.sort()
    for i in range(len(result)):
        result[i]=result[i][1]
    return result

chn = '此开卷第一回也。 作者自云： 因曾历过一番梦幻之后， 故将真事隐去， 而借"通灵\
"之说， 撰此《石头记》 一书也。 故曰"甄士隐"云云． 但书中所记何事何人？ 自又\
云：“今风尘碌碌， 一事无成， 忽念及当日所有之女子， 一一细考较去， 觉其行止见\
识， 皆出于我之上． 何我堂堂须眉， 诚不若彼裙钗哉？ 实愧则有余， 悔又无益之大无可\
如何之日也！ 当此， 则自欲将已往所赖天恩祖德， 锦衣纨绔之时， 饫甘餍肥之日， 背父\
兄教育之恩， 负师友规谈之德， 以至今日一技无成， 半生潦倒之罪， 编述一集， 以告天\
下人： 我之罪固不免， 然闺阁中本自历历有人， 万不可因我之不肖， 自护己短， 一并使\
其泯灭也． 虽今日之茅椽蓬牖， 瓦灶绳床， 其晨夕风露， 阶柳庭花， 亦未有妨我之襟怀\
笔墨者． 虽我未学， 下笔无文， 又何妨用假语村言， 敷演出一段故事来， 亦可使闺阁昭\
传， 复可悦世之目， 破人愁闷， 不亦宜乎？ 故曰"贾雨村"云云。 '
# %% 分词
out1 = jieba.lcut(chn, cut_all=False)
# %% 建立索引
myset = set(out1)  # 利用set去重复
out2 = list(myset) 
out2 = pinyin_sort(out2)  
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