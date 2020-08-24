import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init_classification,
                                           hidden_model, HIDDEN_PARAMS, hidden_init,
                                           feedback_postsyn_model)
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
                                        feedback_wts_model, feedback_wts_init)
import os
import random
import pickle as pkl
from math import ceil
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import minmax_scale


def get_mean_square_error(scale_tr_err_flt, avgsqrerr, get_time, tau_avg_err):
    temp = scale_tr_err_flt * np.mean(avgsqrerr)
    time_in_secs = get_time / 1000
    div = 1.0 - np.exp(-time_in_secs / tau_avg_err) + 1e-9
    error = temp / div
    return error


with open("MNIST_train_spike.pkl", "rb") as f:
    mnist_spike_data = pkl.load(f)


data_dir = "/home/p286814/pygenn/gennzoo_cluster/mnist"
_, y_train = loadlocal_mnist(
    images_path=os.path.join(data_dir, 'train-images-idx3-ubyte'),
    labels_path=os.path.join(data_dir, 'train-labels-idx1-ubyte'))

####### PARAMETERS #########

N_INPUT = 784
N_HIDDEN = 1000
N_OUTPUT = 10
TRIALS = 100
# TRIALS = 60000

TIME_FACTOR = 0.1
update_time = 500
SUPERSPIKE_PARAMS['update_t'] = update_time
SUPERSPIKE_PARAMS['r0'] = 0.1

ITI = mnist_spike_data["iti"]    # ms
WAIT_TIMESTEPS = mnist_spike_data["wait_timesteps"]     # ms
STIMULUS_TIMESTEPS = mnist_spike_data["stimulus_timestpes"]     # ms
WAIT_FREQ = mnist_spike_data["wait_freq"]   # Hz

model_name_build = "mnist"
# MODEL_BUILD_DIR = "./"
MODEL_BUILD_DIR = os.environ.get('TMPDIR') + os.path.sep
IMG_DIR = "/data/p286814/mnist"
# IMG_DIR = "/home/manvi/Documents/gennzoo/mnist"

####### Create training spike train ######

train_poisson_spikes = mnist_spike_data["train_poisson_spikes"]

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

hid = model.add_neuron_population("hid", N_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

out = model.add_neuron_population("out", N_OUTPUT, output_model_classification, OUTPUT_PARAMS, output_init_classification)

inp2hid = model.add_synapse_population("inp2hid", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       inp, hid,
                                       superspike_model, SUPERSPIKE_PARAMS, superspike_init, {}, {},
                                       "ExpCurr", {"tau": 5.0}, {})

hid2out = model.add_synapse_population("hid2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       hid, out,
                                       superspike_model, SUPERSPIKE_PARAMS, superspike_init, {}, {},
                                       "ExpCurr", {"tau": 5.0}, {})

out2hid = model.add_synapse_population("out2hid", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       out, hid,
                                       feedback_wts_model, {}, feedback_wts_init, {}, {},
                                       feedback_postsyn_model, {}, {})

model.build(path_to_model=MODEL_BUILD_DIR)
model.load(path_to_model=MODEL_BUILD_DIR)

########### TRAINING ##############

# Access variables during training time

inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view

hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view
hid_err_tilda = hid.vars['err_tilda'].view

out_voltage = out.vars['V'].view
out_window_of_opp = out.vars["window_of_opp"].view
out_S_pred = out.vars['S_pred'].view
out_S_miss = out.vars['S_miss'].view
out_err_tilda = out.vars['err_tilda'].view

out2hid_wts = out2hid.vars['g'].view

wts_inp2hid = np.array([np.empty(0) for _ in range(N_INPUT * N_HIDDEN)])
wts_hid2out = np.array([np.empty(0) for _ in range(N_HIDDEN * N_OUTPUT)])

time_elapsed = 0

# Random feedback
feedback_wts = np.random.normal(0.0, 1.0, size=(N_HIDDEN, N_OUTPUT)).flatten()
out2hid_wts[:] = feedback_wts
model.push_var_to_device("out2hid", "g")

a = 10.0
b = 5.0
tau_avg_err = 10.0
scale_tr_err_flt = 1.0 / ((((a * b) / (a - b)) ** 2) * (a / 2 + b / 2 - 2 * (a * b) / (a + b))) / tau_avg_err
mul_avgsqrerr = np.exp(-TIME_FACTOR / tau_avg_err)

record_avgsqerr = np.empty(0)
avgsqrerr = np.zeros(shape=N_OUTPUT)
total_time = STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + ITI  # ms

for trial in range(TRIALS):

    if trial % 1 == 0:
        print("\n")
        print("Trial: " + str(trial))

    # Important to record for this trial
    target = y_train[trial]
    t_start = time_elapsed  # ms

    # Reinitialization or providing correct values for diff vars at the start of the next trial
    out_voltage[:] = OUTPUT_PARAMS["Vrest"]
    model.push_var_to_device('out', "V")

    inp_z[:] = ssa_input_init['z']
    model.push_var_to_device("inp", "z")
    inp_z_tilda[:] = ssa_input_init["z_tilda"]
    model.push_var_to_device("inp", "z_tilda")
    hid_z[:] = hidden_init['z']
    model.push_var_to_device("hid", "z")
    hid_z_tilda[:] = hidden_init['z_tilda']
    model.push_var_to_device("hid", "z_tilda")
    hid_voltage[:] = HIDDEN_PARAMS["Vrest"]
    model.push_var_to_device("hid", "V")
    hid2out.vars['lambda'].view[:] = 0.0
    model.push_var_to_device("hid2out", "lambda")
    inp2hid.vars['lambda'].view[:] = 0.0
    model.push_var_to_device("inp2hid", "lambda")
    hid2out.vars['e'].view[:] = 0.0
    model.push_var_to_device("hid2out", "e")
    inp2hid.vars['e'].view[:] = 0.0
    model.push_var_to_device("inp2hid", "e")

    out_err_tilda[:] = 0.0
    model.push_var_to_device('out', 'err_tilda')

    hid_err_tilda[:] = 0.0
    model.push_var_to_device('hid', 'err_tilda')

    out.vars["err_rise"].view[:] = 0.0
    model.push_var_to_device('out', 'err_rise')
    out.vars["err_decay"].view[:] = 0.0
    model.push_var_to_device('out', 'err_decay')

    # Indicate the correct values for window_of_opp, S_pred, and S_miss before the stimulus is presented
    out_window_of_opp[:] = 1.0
    model.push_var_to_device("out", "window_of_opp")

    S_pred = np.zeros(N_OUTPUT)
    S_pred[target] = 1.0
    out.vars['S_pred'].view[:] = S_pred
    model.push_var_to_device("out", "S_pred")

    out_S_miss[:] = 0.0
    model.push_var_to_device("out", "S_miss")

    # Initialize some data structures for recording various things during the trial

    out_V = [np.empty(0) for i in range(N_OUTPUT)]

    out_err = [np.empty(0) for i in range(N_OUTPUT)]

    inp_spike_ids = np.empty(0)
    inp_spike_times = np.empty(0)

    hid_spike_ids = np.empty(0)
    hid_spike_times = np.empty(0)

    produced_spikes = []

    steps = int(total_time / TIME_FACTOR)

    for t in range(steps):

        if t == ((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
            out_window_of_opp[:] = 0.0
            model.push_var_to_device("out", "window_of_opp")

            if len(produced_spikes) == 0:
                out_S_miss[:] = 1.0
                model.push_var_to_device("out", "S_miss")

        model.step_time()

        if t == ((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
            out_S_miss[:] = 0.0
            model.push_var_to_device("out", "S_miss")

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

        # Calculate learning curve
        model.pull_var_from_device("out", "err_tilda")
        temp = out_err_tilda[:]
        temp = np.power(temp, 2)
        temp = np.multiply(temp, TIME_FACTOR)
        avgsqrerr = np.multiply(avgsqrerr, mul_avgsqrerr)
        avgsqrerr = np.add(avgsqrerr, temp)

        if model.t % update_time == 0 and model.t != 0:
            error = get_mean_square_error(scale_tr_err_flt, avgsqrerr, time_elapsed, tau_avg_err)
            record_avgsqerr = np.hstack((record_avgsqerr, error))
            avgsqrerr = np.zeros(shape=N_OUTPUT)

        model.pull_current_spikes_from_device("inp")
        times = np.ones_like(inp.current_spikes) * model.t
        inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
        inp_spike_times = np.hstack((inp_spike_times, times))

        model.pull_current_spikes_from_device("hid")
        times = np.ones_like(hid.current_spikes) * model.t
        hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
        hid_spike_times = np.hstack((hid_spike_times, times))

        model.pull_var_from_device("out", "V")
        new_out_V = out_voltage[:]
        for i in range(len(new_out_V)):
            out_V[i] = np.hstack((out_V[i], new_out_V[i]))

        model.pull_var_from_device("out", "err_tilda")
        new_out_err = out_err_tilda[:]
        for i in range(len(new_out_err)):
            out_err[i] = np.hstack((out_err[i], new_out_err[i]))

    time_elapsed += total_time

    timesteps_plot = np.linspace(t_start, time_elapsed, num=steps)
    num_plots = 4
    fig, axes = plt.subplots(num_plots, sharex=True, figsize=(15, 8))

    for i in range(N_OUTPUT):
        c = "green" if i == target else "red"
        axes[0].plot(timesteps_plot, out_err[i], color=c)
    axes[0].set_title("Error of output neurons")

    for i in range(N_OUTPUT):
        c = "green" if i == target else "red"
        axes[1].plot(timesteps_plot, out_V[i], color=c)
    axes[1].set_title("Membrane voltage of output neurons")
    axes[1].axhline(y=OUTPUT_PARAMS["Vthresh"])
    axes[1].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS,
                    color="green", linestyle="--")

    axes[2].scatter(inp_spike_times, inp_spike_ids)
    axes[2].set_ylim(-1, N_INPUT + 1)
    axes[2].set_title("Input layer spikes")
    axes[2].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS, color="green", linestyle="--")

    axes[3].scatter(hid_spike_times, hid_spike_ids)
    axes[3].set_ylim(-1, N_HIDDEN + 1)
    axes[3].set_title("Hidden layer spikes")

    for i in range(num_plots):
        axes[i].axvspan(t_start, t_start + STIMULUS_TIMESTEPS, facecolor="gray", alpha=0.3)

    axes[-1].set_xlabel("Time [ms]")
    x_ticks_plot = np.linspace(t_start, time_elapsed, 20)
    # x_ticks_plot = list(range(t_start, time_elapsed, int(ceil(5 * TIME_FACTOR))))
    axes[-1].set_xticks(x_ticks_plot)

    save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
    plt.savefig(save_filename)
    plt.close()

save_filename = os.path.join(IMG_DIR, "error.pkl")

with open(save_filename, "wb") as f:
    pkl.dump(record_avgsqerr,f)

print("Complete.")

# # Lambda feedback
# if model.t % 500 == 0 and model.t != 0:
#     # print("Updating feedback weights")
#     model.pull_var_from_device("hid2out", "del_b")
#     del_b = hid2out.vars["del_b"].view[:]
#     model.pull_var_from_device("out2hid", "g")
#     old_fb_wts = out2hid_wts[:]
#     new_fb_wts = np.add(old_fb_wts, del_b)
#     out2hid_wts[:] = new_fb_wts
#     model.push_var_to_device("out2hid", "g")










