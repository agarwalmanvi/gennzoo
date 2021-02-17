from config.plot_config import plot_rcparams
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib import rcParams
import numpy as np

rcParams.update(plot_rcparams)
fig, ax = plt.subplots(figsize=(8,10))
with open('spike_times_arr.pkl', 'rb') as f:
    spike_times_arr = pkl.load(f)
for i in range(len(spike_times_arr)):
    x = np.array(spike_times_arr[i]) - (500*i)
    y = [i] * len(x)
    ax.scatter(x, y)

plt.show()