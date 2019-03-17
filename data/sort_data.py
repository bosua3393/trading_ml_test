import eth_data

data = eth_data.data

size_batch = 300
data_size = len(data)
n_batch = 5000
predict = 24

f = open("eth_sorted.py", "w+")
f.write('batch_x = [[')
for batch in range(n_batch):
    small = data[batch]
    large = data[batch]
    for i in range(size_batch):
        if data[batch + i] < small:
            small = data[batch + i]
        if data[batch + i] > large:
            large = data[batch + i]
    diff = large - small
    for i in range(size_batch):
        element = data[batch + i]
        normalize_element = (element - small) / diff
        f.write("%f" % normalize_element)
        if i == size_batch-1:
            if batch == n_batch - 1:
                f.write(']]')
            else:
                f.write('], [')
        else:
            f.write(', ')


f.write('\n')
f.write('batch_label = [[')
for batch in range(n_batch):
    last_element = data[batch + size_batch]
    future_element = data[batch + size_batch + predict]
    normalize_element = future_element/last_element
    if normalize_element > 1:
        f.write('1, 0')
    else:
        f.write('0, 1')
    if batch == n_batch-1:
        f.write(']]')
    else:
        f.write('], [')

f.close()
