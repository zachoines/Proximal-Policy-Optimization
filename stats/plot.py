import matplotlib.pyplot as plt
import numpy as np

from numpy import arange,array,ones
from scipy import stats


# type what you want to display here
dimension = "CMA"

x, y = np.loadtxt(dimension + ".txt", delimiter=',', unpack=True)
plt.plot(x,y, label='Loaded from file!')

# Best Fit line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept

plt.plot(x, y,'o', x, line)

plt.xlabel('Time')
plt.ylabel(dimension)
plt.title('Statistics View')
plt.legend()
plt.show()