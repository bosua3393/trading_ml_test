import tensorflow as tf
import pandas as pd
import numpy as np
from data import btc_data

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
print(data)

batch_x = data['norm_change'].values
batch_y = data['label'].values


