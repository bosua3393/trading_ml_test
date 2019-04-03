from data import public_client
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

reader = public_client.PublicClient()
live_data = reader.get_product_historic_rates('btc-usd', granularity=3600)
data = [None] * 300

for i in range(300):
    data[i] = live_data[i][1]

plt.plot(np.flip(data))
plt.show()
current = data[0]

min = min(data)
max = max(data)

for i in range(300):
    data[i] = (data[i]-min)/(max-min)

batch_x = [[data]]

model = load_model('keras/0.701600')

r = model.predict(batch_x)

print('Current price: ', current)
print(r)
if r[0][0] > r[0][1]:
    print('Long')
else:
    print('Short')