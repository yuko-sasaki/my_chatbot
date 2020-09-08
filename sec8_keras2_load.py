import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

x = np.linspace(-np.pi, np.pi).reshape(-1, 1)
t = np.sin(x)

model = load_model("model/sinmodel.h5")

plt.plot(x, model.predict(x))
plt.plot(x, t, color="orange")
plt.show()
