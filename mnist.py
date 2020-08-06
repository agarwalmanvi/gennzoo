import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model, OUTPUT_PARAMS, output_init,
                                           hidden_model, HIDDEN_PARAMS, hidden_init,
                                           feedback_postsyn_model)
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
                                        feedback_wts_model, feedback_wts_init,
                                        superspike_reg_model, SUPERSPIKE_REG_PARAMS, superspike_reg_init)
import os
import random
import pickle as pkl
from math import ceil
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import minmax_scale


def to_spike_latency(img_vec, t_eff, thresh):
    img_vec = img_vec.flatten()
    norm_img_vec = minmax_scale(img_vec, feature_range=(0, t_eff - 1))
    spike_img_vec = np.where(norm_img_vec > thresh,
                             t_eff * np.log(norm_img_vec / (norm_img_vec - thresh)),
                             np.inf)
    return spike_img_vec


####### PARAMETERS #########

T_EFF = 50  # ms
THRESH = 0.2
N_INPUT = 784
TIME_FACTOR = 0.1
ITI = 50    # ms

model_name_build = "mnist"

##### Preprocess data into spike time latency format ########

data_dir = "/home/manvi/Documents/pygenn_ml_tutorial/mnist"
X_train, y_train = loadlocal_mnist(
    images_path=os.path.join(data_dir, 'train-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 'train-labels-idx1-ubyte'))

X_train = X_train[:10,:]
X_train_spike = np.zeros(shape=X_train.shape)

# Populate X_train_spike with rows=img_number(60k), cols=N_INPUT(784) and
# the value corresponds to latency to spike time
for img_idx in range(X_train.shape[0]):
    img = X_train[img_idx, :]
    X_train_spike[img_idx, :] = to_spike_latency(img, T_EFF, THRESH)

####### Create training spike train ######

train_intervals = np.amax(X_train_spike, axis=1, where=~np.isinf(X_train_spike), initial=-1)
train_intervals = np.ceil(train_intervals)

add_intervals = np.zeros(X_train_spike.shape[0])
add_intervals[1:] = np.cumsum(train_intervals)[:-1]
# ITI needs to be included
train_poisson_spikes = []

for neuron_idx in range(N_INPUT):
    neuron_spike_times = X_train_spike[:, neuron_idx].flatten()
    neuron_spike_times = np.add(neuron_spike_times, add_intervals)
    train_poisson_spikes.append(neuron_spike_times)

# fig, ax = plt.subplots(figsize=(20,8))
# for i in range(len(train_poisson_spikes)):
#     print("Neuron: " + str(i))
#     y_plot = train_poisson_spikes[i]
#     ax.scatter(y_plot, [i]*len(y_plot), s=10.0)
# for i in range(len(add_intervals)):
#     print("Line for interval: " + str(i))
#     ax.axvline(x=add_intervals[i])
# plt.show()

spike_counts = [len(n) for n in train_poisson_spikes]
train_end_spike = np.cumsum(spike_counts)
train_start_spike = np.empty_like(train_end_spike)
train_start_spike[0] = 0
train_start_spike[1:] = train_end_spike[0:-1]

train_spikeTimes = np.hstack(train_poisson_spikes).astype(float)

########### Custom spike source array neuron model ############

ssa_input_model = genn_model.create_custom_neuron_class(
    "ssa_input_model",
    param_names=["t_rise", "t_decay"],
    var_name_types=[("startSpike", "unsigned int"), ("endSpike", "unsigned int"),
                    ("z", "scalar"), ("z_tilda", "scalar")],
    sim_code="""
    // filtered presynaptic trace
    // $(z) *= exp(- DT / $(t_rise));
    $(z) += (- $(z) / $(t_rise)) * DT;
    $(z_tilda) += ((- $(z_tilda) + $(z)) / $(t_decay)) * DT;
    if ($(z_tilda) < 0.0000001) {
        $(z_tilda) = 0.0;
    }
    """,
    reset_code="""
    $(startSpike)++;
    $(z) += 1.0;
    """,
    threshold_condition_code="$(startSpike) != $(endSpike) && $(t)>= $(spikeTimes)[$(startSpike)]",
    extra_global_params=[("spikeTimes", "scalar*")],
    is_auto_refractory_required=False
)

SSA_INPUT_PARAMS = {"t_rise": 5, "t_decay": 10}

ssa_input_init = {"startSpike": train_start_spike,
                  "endSpike": train_end_spike,
                  "z": 0.0,
                  "z_tilda": 0.0}

########### Build model ################
model = genn_model.GeNNModel("float", model_name_build)
model.dT = 1.0 * TIME_FACTOR

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
inp.set_extra_global_param("spikeTimes", train_spikeTimes)





