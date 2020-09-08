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

print(model.wv.vectors.shape) # 分散表現の形状、(単語数,中間層のニューロン数)
print(model.wv.vectors) # 分散表現

print(len(model.wv.index2word)) # 語彙の数
print(model.wv.index2word[:10]) # 最初の１０単語

print(model.wv.vectors[0]) # 最初のベクトル
print(model.wv.__getitem__("の")) # "の"のベクトル
