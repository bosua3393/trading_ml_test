import eth_data
import matplotlib.pyplot as plt
import numpy


print(eth_data.data)

plot_data = numpy.flip(eth_data.data)

plt.plot(plot_data)
plt.show()
