from data import btc_sorted, btc_data
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('keras_model/0.701600')

r = model.predict([btc_sorted.batch_x])

buy_line = [None] * len(r)
sell_line = [None] * len(r)

for i in range(len(r)):
    if r[i][0] > 0.5:
        if i != 0 and r[i-1][0] < 0.5:
            sell_line[i] = btc_data.data[i]
        buy_line[i] = btc_data.data[i]
    else:
        if i != 0 and r[i-1][0] > 0.5:
            buy_line[i] = btc_data.data[i]
        sell_line[i] = btc_data.data[i]


buy_line = np.flip(buy_line)
sell_line = np.flip(sell_line)
plt.style.use('dark_background')
plt.grid()
plt.plot(buy_line, 'g')
plt.plot(sell_line, 'r')
plt.show()