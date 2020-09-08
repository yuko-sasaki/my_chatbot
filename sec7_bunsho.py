from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle

with open("dataset/ginga_words.pickle", mode='rb') as f:
    ginga_words = pickle.load(f)

tagged_documents = []
for i, sentence in enumerate(ginga_words):
    tagged_documents.append(TaggedDocument(sentence, [i]))

model = Doc2Vec(documents=tagged_documents,
                        size=100,
                        min_count=5,
                        window=5,
                        epochs=20,
                        dm=0)

print(ginga_words[0]) # 最初の文
print(model.docvecs[0]) # 最初の文のベクトル

print(model.docvecs.most_similar(0)) # 最初の文ともっとも類似度が高い文章のベクトル(id, 類似度)

for p in model.docvecs.most_similar(0):
    print(ginga_words[p[0]])
