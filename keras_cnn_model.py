from data import btc_sorted, btc_data
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('keras/0.701600')

r = model.predict([btc_sorted.batch_x])

a = [None]*len(r)
b = [None]*len(r)

for i in range(len(r)):
    if r[i][0] > 0.5:
        a[i] = btc_data.data[i]
    else:
        b[i] = btc_data.data[i]

a = np.flip(a)
b = np.flip(b)
plt.plot(a)
plt.plot(b)
plt.show()