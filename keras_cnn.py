from data import eth_sorted
from data import btc_sorted

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

data_x, data_label = np.array(eth_sorted.batch_x), np.array(eth_sorted.batch_label)
test_x, test_label = np.array(btc_sorted.batch_x), np.array(btc_sorted.batch_label)

model = Sequential()

model.add(Dense(512, input_shape=(300,), activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

history = model.fit(data_x, data_label, validation_data=(test_x, test_label), epochs=500)

acc_log = history.history['val_acc']

last_acc = acc_log[len(acc_log)-1]

model.save('keras/%f' % last_acc)
