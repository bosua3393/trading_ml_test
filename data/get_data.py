from public_client import PublicClient
import datetime as DT
import time

batch_size = 300
candle = 3600
n_batch = 50

start_time = 1552717954

reader = PublicClient()
# f = open('eth_data.py', 'w+')
for i in range(n_batch):
    batch_end = DT.datetime.utcfromtimestamp(start_time - (i * candle * batch_size)).isoformat()
    batch_start = DT.datetime.utcfromtimestamp(start_time - ((i + 1) * candle * batch_size)).isoformat()
    data = reader.get_product_historic_rates('eth-usd', start=str(batch_start), end=str(batch_end), granularity=candle)
    time.sleep(1)
    for step in range(len(data)):
        f.write('%f ,' % (data[step][1]))

f.close()
