from janome.tokenizer import Tokenizer

t = Tokenizer()

s = "すもももももももものうち"

for token in t.tokenize(s):
    # <class 'janome.tokenizer.Token'>
    print(token)# すもも  名詞,一般,*,*,*,*,すもも,スモモ,スモモ
    # <class 'str'>
    print(token.surface)# すもも
    print(token.part_of_speech)# 名詞,一般,*,*
    print(token.infl_type)# *
    print(token.infl_form)# *
    print(token.base_form)# すもも
    print(token.reading)# スモモ
    print(token.phonetic)# スモモ
    print(token.node_type)# SYS_DICT

word_list = t.tokenize(s, wakati=True)
#print(word_list)# <generator object Tokenizer.__tokenize_stream at 0x000001F3EEA0EDB0>

word_list = [ token.surface for token in t.tokenize(s)]
#print(word_list)# ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち']

def make_wakati_list(sentencex):
    return [ token.surface for token in t.tokenize(sentencex)]

print(make_wakati_list(s))

import pickle
import collections

with open("dataset/ginga_list.pickle", mode='rb') as f:
    ginga_list = pickle.load(f)
ginga_words = []
for ginga_sentence in ginga_list:
    ginga_words.append(make_wakati_list(ginga_sentence))

with open("dataset/ginga_words.pickle", mode='wb') as f:
    pickle.dump(ginga_words, f)

with open("dataset/ginga_words.pickle", mode='rb') as f:
    ginga_words = pickle.load(f)
print(ginga_words)
print(type(ginga_words))

"""
c = collections.Counter(ginga_words)
with open("dataset/ginga_words_c.pickle", mode='wb') as f:
    pickle.dump(c, f)
"""
