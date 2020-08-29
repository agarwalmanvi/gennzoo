from utils import create_poisson_spikes
import os
from mlxtend.data import loadlocal_mnist
import pickle as pkl
import numpy as np
from sklearn.preprocessing import minmax_scale


def to_spike_latency(img_vec, t_eff, thresh):
    img_vec = img_vec.flatten()
    norm_img_vec = minmax_scale(img_vec, feature_range=(0, t_eff - 1))
    spike_img_vec = np.where(norm_img_vec > thresh,
                             t_eff * np.log(norm_img_vec / (norm_img_vec - thresh)),
                             np.inf)
    return spike_img_vec


"""
In the following script, we will convert the samples from the MNIST dataset into
a spike latency dataset. This means that
for every sample, the pixel values will be converted into spike times
such that each of the 784 input neurons spike exactly once during
an input presentation.
We will use the method described in the paper:
The remarkable robustness of surrogate gradient learning for instilling complex function in spiking neural networks
by Friedemann Zenke and Tim P. Vogels
"""

T_EFF = 50  # ms
THRESH = 0.2

data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
X_train, y_train = loadlocal_mnist(
    images_path=os.path.join(data_dir, 'train-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 'train-labels-idx1-ubyte'))

"""
Here we initialize a data structure to store the spike times for each pixel for each sample.
We will populate `X_train_spike`, of size rows=60,000 and cols=784,
with values corresponding to latency to spike time for each pixel.
Note that many values in this data structure will be `np.inf`
meaning that these input neurons should not spike during sample presentation.
Finally, we would like to dump the results in a pickle file so that
we can use it later for many scripts, using different combinations of
sample presentations.
"""
X_train_spike = np.zeros(shape=X_train.shape)

# Populate X_train_spike with rows=img_number(60k), cols=N_INPUT(784) and
# the value corresponds to latency to spike time for each pixel
for img_idx in range(X_train.shape[0]):
    if img_idx % 10 == 0:
        print("Processing img " + str(img_idx))
    img = X_train[img_idx, :]
    X_train_spike[img_idx, :] = to_spike_latency(img, T_EFF, THRESH)

X_train_spike = np.round(X_train_spike, decimals=1)

with open("MNIST_train_spike.pkl", "wb") as f:
    pkl.dump(X_train_spike, f)

"""
We can also do this for the testing dataset and store it as a pickle file for multiple scripts
without having to generate the spike latency dataset over and over again.
"""