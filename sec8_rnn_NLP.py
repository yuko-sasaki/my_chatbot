import re

with open("dataset/gingatetsudono_yoru.txt", mode='r', encoding="utf-8") as f:
    ginga_original = f.read()

ginga = re.sub("《[^《]+》", "", ginga_original)
ginga = re.sub("※［[^［]+］", "", ginga)
ginga = re.sub("［[^［]+］[^［]+［[^［]+］", "", ginga)
ginga = re.sub("〔以下[^〔]+〕", "", ginga)
ginga = re.sub("〔[^〔]+空白〕", "・・", ginga)
ginga = re.sub("……", "……。", ginga)
ginga = re.sub("。。", "。", ginga)
text = re.sub("[ 　「」〔〕（）\n]", "", ginga)

n_rnn = 10 # 時系列の数
batch_size = 128
epochs = 60
n_mid = 128

# make one hot vec
import numpy as np

chars = sorted(list(set(text)))
print("文字数：" + str(len(chars)))
char_indices = {}
for i, char_ in enumerate(chars):
    char_indices[char_] = i
indices_char = {}
for i, char_ in enumerate(chars):
    indices_char[i] = char_

time_chars = []
next_chars = []
for i in range(0, len(text) - n_rnn):
    time_chars.append(text[i:i+n_rnn])
    next_chars.append(text[i+n_rnn])

x = np.zeros((len(time_chars), n_rnn, len(chars)), dtype=np.bool)
t = np.zeros((len(time_chars), len(chars)), dtype=np.bool)
for i, t_cs in enumerate(time_chars):
    t[i, char_indices[next_chars[i]]] = 1
    for j, char_ in enumerate(t_cs):
        x[i, j, char_indices[char_]] = 1
print("x shape")
print(x.shape)
print("t shape")
print(t.shape)

# rnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(n_mid, input_shape=(n_rnn, len(chars))))
model.add(Dense(len(chars),activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
print(model.summary())

# 文書生成関数
from tensorflow.keras.callbacks import LambdaCallback

def on_epoch_end(epoch, logs):
    beta = 5
    prev_text = text[0:n_rnn]
    created_text = prev_text

    print("シード：" + created_text)

    for i in range(400):
        x_pred = np.zeros((1, n_rnn, len(chars)))
        for j, char_ in enumerate(prev_text):
            x_pred[0, j, char_indices[char_]] = 1

        y = model.predict(x_pred)
        p_power = y[0] ** beta
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power)) # pは確率分布、pに従って次の文字が取り出される
        next_char = indices_char[next_index]

        created_text += next_char
        prev_text = prev_text[1:] + next_char
    print(created_text)
    print()

epock_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# 学習
history = model.fit(x, t, epochs=20, batch_size=batch_size, callbacks=[epock_end_callback])
