import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

x = np.linspace(-np.pi, np.pi).reshape(-1, 1)
t = np.sin(x)

batch_size = 8
n_in = 1
n_mid = 20
n_out = 1

model = Sequential()
model.add(Dense(n_mid, input_shape=(n_in,), activation="sigmoid"))# 活性化関数：シグモイド関数
model.add(Dense(n_out,activation="linear"))# 活性化関数：恒等関数
model.compile(loss="mean_squared_error", optimizer="sgd")# 損失関数：二条誤差、最適化アルゴリズム：SGD
print(model.summary())

# 学習
history = model.fit(x, t, batch_size=batch_size, epochs=2000, validation_split=0.1)

loss = history.history['loss'] # 訓練用データの誤差
vloss = history.history['val_loss'] # 検証用データの誤差

plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(vloss)), vloss, color="orange")
plt.show()

model.save("model/sinmodel.h5")
