from data import eth_sorted
from data import btc_sorted

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
import numpy as np

data_x, data_label = np.array(eth_sorted.batch_x), np.array(eth_sorted.batch_label)
test_x, test_label = np.array(btc_sorted.batch_x), np.array(btc_sorted.batch_label)

data_x = data_x.reshape(1, 5000, 300)
data_label = data_label.reshape(1, 5000, 2)
test_x = test_x.reshape(1, 5000, 300)
test_label = test_label.reshape(1, 5000, 2)

model = Sequential()

model.add(LSTM(128, input_shape=(5000, 300), return_sequences=True))

model.add(LSTM(128, input_shape=(5000, 300), return_sequences=True))

model.add(TimeDistributed(Dense(2, activation='softmax')))

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

history = model.fit(data_x, data_label, validation_data=(test_x, test_label), epochs=200)

acc_log = history.history['val_acc']

last_acc = acc_log[len(acc_log)-1]

model.save('keras_model/rnn_%f' % last_acc)
