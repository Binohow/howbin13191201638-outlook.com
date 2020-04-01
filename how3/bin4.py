""" 
情感分析实例：IMDB 影评情感分析 
"""
# %%
from keras.datasets import imdb
import numpy as np
from matplotlib import pyplot as plt
# %% 导入数据
(x_train, y_train), (x_test, y_test) = imdb.load_data()
# 合并训练数据集和评估数据集
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

print('x shape is %s, y shape is %s' % (x.shape, y.shape))
print('Classes: %s' % np.unique(y))
print('Total words: %s' % len(np.unique(np.hstack(x))))

result = [len(word) for word in x]
print('Mean: %.2f words (STD: %.2f)' %(np.mean(result), np.std(result)))
# 图表展示
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()