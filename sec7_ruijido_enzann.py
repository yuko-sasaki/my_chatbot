from gensim.models import word2vec
import pickle

with open("dataset/ginga_words.pickle", mode='rb') as f:
    ginga_words = pickle.load(f)

model = word2vec.Word2Vec(ginga_words,
                        size=100,
                        min_count=5,
                        window=5,
                        iter=20,
                        sg=0)

print(model.wv.most_similar("銀河")) # コサイン類似度

import numpy as np

a = model.wv.__getitem__("銀河")
b = model.wv.__getitem__("立派")

cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) # コサイン類似度
print(cos_sim)

print(model.wv.most_similar(positive=["銀河", "天の川"])) # 銀河 + 天の川

print(model.wv.most_similar(positive=["銀河", "天の川"], negative=["みんな"])) # 銀河 + 天の川 - みんな
