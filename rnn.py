import pandas as pd
import numpy as np
from data import btc_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributedDense
from tensorflow.keras import optimizers


FUTURE = 24


def label(current, future):
    if future > current:
        return 1
    else:
        return 0


data = pd.DataFrame({'btc_usd': np.flip(btc_data.data)})
data['norm_change'] = 1-(data['btc_usd'].shift(1)/data['btc_usd'])
data['norm_change'] = data['norm_change']/(abs(data['norm_change']).max())
data['future'] = data['btc_usd'].shift(-FUTURE)
data = data[:-FUTURE]
data['label'] = list(map(label, data['btc_usd'], data['future']))
data = data.dropna()

batch_x = np.array(data['norm_change'].values)
batch_y = np.array(data['label'].values)

batch_x = batch_x.reshape(1, 14949, 1)
batch_y = batch_y.reshape(1, 14949, 1)

print(batch_x, batch_y)


model = Sequential()

model.add(LSTM(128, input_shape=(14949, 1), return_sequences=True))

model.add

opt = optimizers.Adam(lr=0.0001, decay=.000001)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')

model.fit(batch_x, batch_y, epochs=5)
