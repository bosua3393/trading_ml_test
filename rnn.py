import pandas as pd
import numpy as np
from data import btc_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras import optimizers

FUTURE = 24
length = 14949


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

batch_x = batch_x.reshape(1, length, 1)
print(batch_x)
batch_y = batch_y.reshape(1, length, 1)

model = Sequential()

model.add(LSTM(128, input_shape=(length, 1), return_sequences=True))

model.add(TimeDistributed(Dense(1, activation='softmax')))

opt = optimizers.Adam(lr=.001, decay=.000001)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
'''
model.fit(batch_x, batch_y, epochs=5, verbose=1)

result = model.predict(batch_x)
for i in range(len(result[0])):
    print(result[0][i])
'''