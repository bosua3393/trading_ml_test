import pandas as pd
import numpy as np
from data import btc_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras import optimizers

FUTURE = 24
x_length = 24
length = 14949 - x_length


def data_processing(x, y, x_length):
    time_step = []
    seq = []
    seq_label = []
    for i in range(len(x)):
        time_step.insert(0, x[i])
        if i >= x_length:
            time_step.pop()
            seq.append(time_step[:])
            seq_label.append([y[i]])
    return np.array([seq]), np.array([seq_label])


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

batch_x, batch_y = data_processing(data['norm_change'].values, data['label'].values, x_length)
#print(batch_x, batch_y)

model = Sequential()

model.add(LSTM(32, input_shape=(length, x_length), return_sequences=True, activation='sigmoid'))

model.add(LSTM(32, input_shape=(length, x_length), return_sequences=True, activation='sigmoid'))

model.add(TimeDistributed(Dense(12, activation='sigmoid')))

model.add(TimeDistributed(Dense(1, activation='softmax')))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

model.fit(batch_x, batch_y, epochs=2, verbose=1)

result = model.predict(batch_x)
for i in range(len(result[0])):
    print(result[0][i])
