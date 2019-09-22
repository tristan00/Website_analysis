# from gensim.models.keyedvectors import KeyedVectors
# import gensim
# import os
# import shutil
# import hashlib
# from sys import platform
#
#
# def getFileLineNums(filename):
#     f = open(filename, 'r')
#     count = 0
#     for line in f:
#         count += 1
#     return count
#
#
# def prepend_line(infile, outfile, line):
#     with open(infile, 'r') as old:
#         with open(outfile, 'w') as new:
#             new.write(str(line) + "\n")
#             shutil.copyfileobj(old, new)
#
# def prepend_slow(infile, outfile, line):
#     with open(infile, 'r') as fin:
#         with open(outfile, 'w') as fout:
#             fout.write(line + "\n")
#             for line in fin:
#                 fout.write(line)
#
# def load(filename):
#     num_lines = getFileLineNums(filename)
#     gensim_file = 'glove_model.txt'
#     gensim_first_line = "{} {}".format(num_lines, 300)
#     # Prepends the line.
#     if platform == "linux" or platform == "linux2":
#         prepend_line(filename, gensim_file, gensim_first_line)
#     else:
#         prepend_slow(filename, gensim_file, gensim_first_line)
#
#     model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
#     return model
# model = load("/home/td/Downloads/glove.840B.300d.txt")
#
#
# print('deliveries', 'delivery', model.similarity('deliveries', 'delivery'))
# print('restaurant', 'delivery', model.similarity('restaurant', 'delivery'))
# print('hospital', 'delivery', model.similarity('hospital', 'delivery'))

# print(ord(u'h'))

# path = '/home/td/Downloads/AOL-user-ct-collection'
#
# import glob
# import pandas as pd
#
# files = glob.glob(f'{path}/*.txt')
# dfs = [pd.read_csv(i, sep='\t') for i in files]
# df = pd.concat(dfs)
# df = df.dropna(subset=['ClickURL'])
# df = df.drop_duplicates(subset=['ClickURL'])
# df = df[['ClickURL']]
# df.to_csv('/tmp/web_crawler_input.txt')


# import numpy as np
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[2, 2], [2, 2]])
# c = a - b
# print(c)


# from bert_embedding import BertEmbedding
# a = ['the dog and the cat']
# bert = BertEmbedding()
# res = bert(a)
#
# print(len(res))
# print(len(res[0]))
# print(len(res[0][0]))
# print(len(res[0][1]))

from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamodel
import string

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def tokenize(s):
    return str(s).lower().translate(translator).split()

documents = ['the cat in the hat',
             'sports are fun',
             'phones suck',
             'cars really suck',
             'i fucked up this morning']


documents_tokenized = [tokenize(i) for i in documents]
common_dictionary = Dictionary(documents_tokenized)
common_corpus = [common_dictionary.doc2bow(text) for text in documents_tokenized]
vectorizer = ldamodel.LdaModel(common_corpus, id2word=common_dictionary, num_topics=2)


other_corpus = [common_dictionary.doc2bow(tokenize(i)) for i in documents]


for i in other_corpus:
    print(vectorizer[i])

