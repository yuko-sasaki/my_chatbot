import re
import pickle

with open("dataset/gingatetsudono_yoru.txt", mode='r', encoding="utf-8") as f:
    ginga_original = f.read()

ginga = re.sub("《[^《]+》", "", ginga_original)
ginga = re.sub("※［[^［]+］", "", ginga)
ginga = re.sub("［[^［]+］[^［]+［[^［]+］", "", ginga)
ginga = re.sub("〔以下[^〔]+〕", "", ginga)
ginga = re.sub("〔[^〔]+空白〕", "・・", ginga)
ginga = re.sub("……", "……。", ginga)
ginga = re.sub("。。", "。", ginga)
ginga = re.sub("[ 　「」〔〕（）\n]", "", ginga)

seperator = "。"
ginga_list = ginga.split(seperator)
ginga_list.pop()
ginga_list = [ ginga_text + seperator for ginga_text in ginga_list]


with open("dataset/ginga_list.pickle", mode='wb') as f:
    pickle.dump(ginga_list, f)

with open("dataset/ginga_list.pickle", mode='rb') as f:
    ginga_list = pickle.load(f)

print(ginga_list)
