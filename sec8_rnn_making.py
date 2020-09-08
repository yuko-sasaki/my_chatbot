import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-2*np.pi, 2*np.pi)
sin_data = np.sin(x_data) + 0.1*np.random.randn(len(x_data)) # ノイズ追加

#plt.plot(x_data, sin_data)
#plt.show()

n_rnn = 10 # 時系列の数
n_sample = len(x_data) - n_rnn # サンプル数
x = np.zeros((n_sample, n_rnn)) # 入力
t = np.zeros((n_sample, n_rnn)) # 正解
for i in range(0, n_sample):
    x[i] = sin_data[i:i+n_rnn]
    t[i] = sin_data[i+1:i+n_rnn+1] # 時系列を入力よりも１つずらす

x = x.reshape(n_sample, n_rnn, 1) # kerasのPNNでは入力を（サンプル数、時系列の数、入力層ニューロン数）
print(x.shape)
t = t.reshape(n_sample, n_rnn, 1)
print(t.shape)


# RNN構築
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

batch_size = 8
n_in = 1
n_mid = 20
n_out = 1

model = Sequential()
model.add(SimpleRNN(n_mid, input_shape=(n_rnn, n_in,), return_sequences=True))
model.add(Dense(n_out,activation="linear"))# 活性化関数：恒等関数
model.compile(loss="mean_squared_error", optimizer="sgd")# 損失関数：二条誤差、最適化アルゴリズム：SGD
print(model.summary())

history = model.fit(x, t, epochs=20, batch_size=batch_size, validation_split=0.1)

loss = history.history['loss'] # 訓練用データの誤差
vloss = history.history['val_loss'] # 検証用データの誤差

plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(vloss)), vloss, color="orange")
plt.show()

predicted = x[0].reshape(-1)

for i in range(0, n_sample):
    y = model.predict(predicted[-n_rnn:].reshape(1, n_rnn, 1))
    predicted = np.append(predicted, y[0][n_rnn-1][0])

plt.plot(np.arange(len(sin_data)), sin_data, label="training data")
plt.plot(np.arange(len(predicted)), predicted, label="predicted")
plt.legend()
plt.show()
