import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.preprocessing.text as text
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
txt1 = "Those that come through the Ivory Gate cheat us with empty promises \
that never see fulfillment. Those that come through the Gate of Horn inform \
the dreamer of the truth"
txt2='"It is impossible to describe our pain," the family said, adding that his death appeared to have occurred before the coronavirus pandemic.\
Levinson, the longest-held hostage in US history, disappeared in 2007 on the Iranian island of Kish.\
His family say he was working on behalf of an unauthorised CIA mission.'
alltext = [txt1, txt2]

#tokenizer = Tokenizer(nb_words=1000)
#tokenizer.fit_on_texts(alltext)

texts = ['The cat sat on the mat.',
         'The dog sat on the log.',
         'Dogs and cats living together.']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print('word_counts: ', tokenizer.word_counts)
print('word_docs: ', tokenizer.word_docs)
print('word_index: ', tokenizer.word_index)
print('document_count: ', tokenizer.document_count)

sequences = tokenizer.texts_to_sequences(texts)
tokenizer.fit_on_sequences(sequences)

for mode in ['binary', 'count', 'tfidf', 'freq']:
    matrix = tokenizer.texts_to_matrix(texts, mode)
    print(mode, " : ", matrix)