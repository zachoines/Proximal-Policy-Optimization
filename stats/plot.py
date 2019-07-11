import matplotlib.pyplot as plt
import numpy as np

from numpy import arange,array,ones
from scipy import stats
from heapq import *


# type what you want to display here
dimension = "LOSS"
heap = []

x, runtime, y = np.loadtxt(dimension + ".txt", delimiter=',', unpack=True)

for i in range(len(x)):
    data = y[i]
    time = x[i]
    item = (time, data)
    heappush(heap, item)

    
sorted_x = []
sorted_y = []
counter = 0
while heap:
    (x_i, y_i) = heappop(heap)
    sorted_x.append(counter)
    sorted_y.append(y_i)
    counter += 1


# TODO::Statistically find outliers in a data set and exclude then from. IQR
sorted_x = np.array(sorted_x, dtype = int)

# Best Fit line
slope, intercept, r_value, p_value, std_err = stats.linregress(sorted_x, sorted_y)
line = slope * sorted_x + intercept
plt.plot(sorted_x, sorted_y, label='Slope: ' + str(slope))

plt.plot(sorted_x, sorted_y,'o', sorted_x, line)

plt.xlabel('Time')
plt.ylabel(dimension)
plt.title('Statistics View')
plt.legend()
plt.show()