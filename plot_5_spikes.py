from config.plot_config import plot_rcparams
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib import rcParams
import numpy as np
import os

rcParams.update(plot_rcparams)
fig, ax = plt.subplots(figsize=(8,6))
with open('spike_times_arr.pkl', 'rb') as f:
    spike_times_arr = pkl.load(f)
for i in range(len(spike_times_arr)):
    x = np.array(spike_times_arr[i]) - (500*i)
    y = [i] * len(x)
    ax.scatter(x, y, c="black", s=12)
# with open('avgsqerr.pkl', 'rb') as f:
#     avgsqerr = pkl.load(f)
# x = list(range(avgsqerr.shape[1]))
# for i in range(avgsqerr.shape[0]):
#     ax.plot(x, avgsqerr[i, :], color="gray", alpha=0.4)
# mean_loss = np.mean(avgsqerr, axis=0)
# ax.plot(x, mean_loss, color="black", alpha=0.7, linestyle="dashed")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Trial")
# ax.set_xlabel("Time [trials]")
# ax.set_ylabel("Cost")
fig.tight_layout()

IMG_DIR = "imgs"
save_filename = os.path.join(IMG_DIR, "spike_times_arr.png")
# save_filename = os.path.join(IMG_DIR, "avgsqerr.png")
plt.savefig(save_filename)
plt.close()
