import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model, OUTPUT_PARAMS, output_init,
                                           hidden_model, HIDDEN_PARAMS, hidden_init,
                                           feedback_postsyn_model)
from models.neurons.ssa_input import ssa_input_model, ssa_input_init, SSA_INPUT_PARAMS
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
                                        feedback_wts_model, feedback_wts_init)
import os
from models.parameters import r0, tau_avg_err
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from utils import create_poisson_spikes
from config.plot_config import plot_rcparams

rcParams.update(plot_rcparams)

"""
In this script, we are going to recreate the experiment from Fig. 4 (a,b) of the Superspike paper.
This is an extension of the experiment from the `timed_spikes_exp1.py` file.
In this experiment, one output neuron is trained to output
5 equidistant spikes in response to a ms repeated noise Poisson spike train
from 100 input neurons. In addition, we also have a hidden layer between the input
and output layers, with 4 hidden neurons.
"""

PRESENT_TIMESTEPS = 500.0
TRIALS = 1600
# TRIALS = 10
NUM_HIDDEN = 4
SUPERSPIKE_PARAMS["update_t"] = PRESENT_TIMESTEPS
TIME_FACTOR = 0.1

"""
First we create the poisson spike trains for all the input neurons.
Below, `poisson_spikes` is a list of 100 lists: each list is the spike times for each neuron.
We create `start_spike` and `end_spike`, which we need to initialize the input layer.
`start_spike` and `end_spike` give the indices at which each neuron's spike times starts and ends
e.g. start_spike[0] is the starting index and end_spike[0] is the ending index of the 0th neuron's spike times.
"""

poisson_spikes = []
freq = 8
spike_dt = 0.001
N_INPUT = 100
interval = int(PRESENT_TIMESTEPS)
for p in range(N_INPUT):
    neuron_spike_train = create_poisson_spikes(interval, freq, spike_dt, 1.0)
    poisson_spikes.append(neuron_spike_train)

spike_counts = [len(n) for n in poisson_spikes]

end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

spikeTimes = np.hstack(poisson_spikes).astype(float)

"""
The target spike train is a series of 5 equidistant spikes, which we create below.
"""

base_target_spike_times = np.linspace(0, 500, num=7)[1:6].astype(int)
add_arr = np.arange(TRIALS)
add_arr = np.repeat(add_arr, len(base_target_spike_times))
add_arr = np.multiply(add_arr, interval)
target_spike_times = np.tile(base_target_spike_times, TRIALS)
target_spike_times = np.add(target_spike_times, add_arr)

target_poisson_spikes = [target_spike_times]
spike_counts = [len(n) for n in target_poisson_spikes]
target_end_spike = np.cumsum(spike_counts)
target_start_spike = np.empty_like(target_end_spike)
target_start_spike[0] = 0
target_start_spike[1:] = target_end_spike[0:-1]

target_spikeTimes = np.hstack(target_poisson_spikes).astype(float)

# Set up startSpike and endSpike for custom SpikeSourceArray model

ssa_input_init["startSpike"] = start_spike
ssa_input_init["endSpike"] = end_spike

########### Build model ################
model = genn_model.GeNNModel("float", "spike_source_array", time_precision="double")
model.dT = 1.0 * TIME_FACTOR

inp = model.add_neuron_population("inp", 100, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
inp.set_extra_global_param("spikeTimes", spikeTimes)

hid = model.add_neuron_population("hid", NUM_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

output_init['startSpike'] = target_start_spike
output_init['endSpike'] = target_end_spike
out = model.add_neuron_population("out", 1, output_model, OUTPUT_PARAMS, output_init)
out.set_extra_global_param("spikeTimes", target_spikeTimes)

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

model.build()
model.load()

"""
We use Xavier initialization to set the weights of the inp2hid and hid2out connections
"""
a = 1.0 / np.sqrt(N_INPUT)
wt_init = np.random.uniform(low=-a, high=a, size=(N_INPUT * NUM_HIDDEN)).flatten()
inp2hid.vars["w"].view[:] = wt_init
model.push_var_to_device("inp2hid", "w")
a = 1.0 / np.sqrt(NUM_HIDDEN)
wt_init = np.random.uniform(low=-a, high=a, size=NUM_HIDDEN)
hid2out.vars["w"].view[:] = wt_init
model.push_var_to_device("hid2out", "w")

"""
Here we will use random feedback. The feedback weights need to be set and pushed
to the model only once at the start of the simulation, which we do below.
"""
feedback_wts = np.random.normal(0.0, 1.0, size=(NUM_HIDDEN, 1)).flatten()
out2hid.vars['g'].view[:] = feedback_wts
model.push_var_to_device('out2hid', 'g')

IMG_DIR = "imgs_exp2"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

spikeTimes_view = inp.extra_global_params['spikeTimes'].view
start_spike_view = inp.vars['startSpike'].view
out_err_tilda = out.vars['err_tilda'].view
out_err_rise = out.vars["err_rise"].view
out_err_decay = out.vars["err_decay"].view
out_voltage = out.vars['V'].view
inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view
hid_err_tilda = hid.vars['err_tilda'].view
hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view
inp2hid_lambda = inp2hid.vars['lambda'].view
hid2out_lambda = hid2out.vars['lambda'].view
inp2hid_e = inp2hid.vars['e'].view
hid2out_e = hid2out.vars['e'].view

"""
Each plot will have:
1. Spikes of input population
2. Membrane potential of hidden neurons
3. Error of output neuron
4. Target spike train
5. Membrane potential of output neuron
"""

plot_interval = 1
base_timesteps = np.arange(int(PRESENT_TIMESTEPS), step=1.0*TIME_FACTOR)

print("r0 at the start: " + str(r0))

"""
We also create all the ingredients needed to calculate the error
"""
a = 10.0
b = 5.0
scale_tr_err_flt = 1.0 / ((((a * b) / (a - b)) ** 2) * (a / 2 + b / 2 - 2 * (a * b) / (a + b))) / tau_avg_err
record_avgsqerr = np.empty(0)

steps_in_trial = int(PRESENT_TIMESTEPS / TIME_FACTOR)

for trial in range(TRIALS):

    # Decrease the learning rate every 600th trial
    if trial != 0 and trial % 600 == 0:
        r0 *= 0.1
        inp2hid.vars["r0"].view[:] = r0
        model.push_var_to_device('inp2hid', "r0")
        hid2out.vars["r0"].view[:] = r0
        model.push_var_to_device('hid2out', "r0")
        print("Changed r0 to: " + str(r0))

    # Reset variables at the beginning of a trial
    out_voltage[:] = OUTPUT_PARAMS["Vrest"]
    model.push_var_to_device('out', "V")
    inp_z[:] = ssa_input_init['z']
    model.push_var_to_device("inp", "z")
    inp_z_tilda[:] = ssa_input_init["z_tilda"]
    model.push_var_to_device("inp", "z_tilda")
    out_err_tilda[:] = 0.0
    model.push_var_to_device('out', 'err_tilda')
    out_err_rise[:] = 0.0
    model.push_var_to_device('out', 'err_rise')
    out_err_decay[:] = 0.0
    model.push_var_to_device('out', 'err_decay')
    hid_z[:] = hidden_init['z']
    model.push_var_to_device("hid", "z")
    hid_z_tilda[:] = hidden_init['z_tilda']
    model.push_var_to_device("hid", "z_tilda")
    hid_voltage[:] = HIDDEN_PARAMS["Vrest"]
    model.push_var_to_device("hid", "V")
    hid_err_tilda[:] = 0.0
    model.push_var_to_device('hid', 'err_tilda')

    hid2out_lambda[:] = 0.0
    model.push_var_to_device("hid2out", "lambda")
    inp2hid_lambda[:] = 0.0
    model.push_var_to_device("inp2hid", "lambda")
    hid2out_e[:] = 0.0
    model.push_var_to_device("hid2out", "e")
    inp2hid_e[:] = 0.0
    model.push_var_to_device("inp2hid", "e")

    produced_spike_train = []

    if trial % 10 == 0:
        print("Trial: " + str(trial))

    # Initialize data structures to store various things during a trial so we can make a nice plot later
    if trial % plot_interval == 0:
        spike_ids = np.empty(0)
        spike_times = np.empty(0)
        error = np.empty(0)
        out_V = np.empty(0)
        hidden_V = [np.empty(0) for _ in range(NUM_HIDDEN)]
        hidden_spikes = [np.empty(0) for _ in range(NUM_HIDDEN)]

    if trial != 0:
        spikeTimes += PRESENT_TIMESTEPS

        spikeTimes_view[:] = spikeTimes
        model.push_extra_global_param_to_device("inp", "spikeTimes")

        start_spike_view[:] = start_spike
        model.push_var_to_device("inp", "startSpike")

    for t in range(steps_in_trial):

        model.step_time()

        if trial % plot_interval == 0:

            model.pull_current_spikes_from_device("inp")
            times = np.ones_like(inp.current_spikes) * model.t
            spike_ids = np.hstack((spike_ids, inp.current_spikes))
            spike_times = np.hstack((spike_times, times))

            model.pull_current_spikes_from_device("out")
            if len(out.current_spikes) != 0:
                produced_spike_train.append(model.t)

            model.pull_var_from_device("out", "err_tilda")
            error = np.hstack((error, out.vars["err_tilda"].view))

            model.pull_var_from_device("out", "V")
            out_V = np.hstack((out_V, out.vars["V"].view))

            model.pull_var_from_device("hid", "V")
            new_hidden_V = hid.vars["V"].view
            for i in range(len(hidden_V)):
                hidden_V[i] = np.hstack((hidden_V[i], new_hidden_V[i]))

            model.pull_current_spikes_from_device("hid")
            for i in hid.current_spikes:
                hidden_spikes[i] = np.hstack((hidden_spikes[i], model.t))

    # create a pretty plot
    if trial % plot_interval == 0:

        timesteps = base_timesteps + int(PRESENT_TIMESTEPS * trial)

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(nrows=4+4, ncols=1, height_ratios=[0.5, 2, 2, 2, 0.5, 0.5, 0.5, 0.5])
        axes = []
        for i in range(4+4):
            if i == 0:
                axes.append(fig.add_subplot(gs[i]))
            else:
                axes.append(fig.add_subplot(gs[i], sharex=axes[0]))
        plt.subplots_adjust(hspace=1.0)

        target_spike_times_plot = base_target_spike_times + int(PRESENT_TIMESTEPS * trial)

        axes[0].scatter(target_spike_times_plot, [1] * len(target_spike_times_plot), s=55)
        axes[0].set_title("Target spike train")
        axes[0].set_xlim(timesteps[0], timesteps[-1] + 1)
        axes[0].axes.get_yaxis().set_visible(False)
        axes[0].axes.get_xaxis().set_visible(False)
        [s.set_visible(False) for s in axes[0].spines.values()]

        axes[1].plot(timesteps, error, linewidth=2.5)
        axes[1].set_title("Error")
        axes[1].axes.get_xaxis().set_visible(False)
        axes[1].axes.get_yaxis().set_visible(False)
        [s.set_visible(False) for s in axes[1].spines.values()]

        axes[2].plot(timesteps, out_V, linewidth=2.5)
        axes[2].axhline(y=OUTPUT_PARAMS["Vthresh"], linestyle="--", color="red", linewidth=3)
        if trial == 0:
            axes[2].text(x=timesteps[0] + 30, y=-52, s=r"$U_{thresh}$", fontsize=20,
                         horizontalalignment='center', verticalalignment='center',
                         bbox=dict(facecolor='white', edgecolor="white"), color="red")
        axes[2].set_title("Membrane potential of output neuron")
        for i in produced_spike_train:
            axes[2].axvline(x=i, linestyle="dotted", color="red", linewidth=3, alpha=0.6)
        axes[2].axes.get_xaxis().set_visible(False)
        axes[2].axes.get_yaxis().set_visible(False)
        [s.set_visible(False) for s in axes[2].spines.values()]

        axes[3].scatter(spike_times, spike_ids, s=20)
        axes[3].set_title("Input spikes")
        [s.set_visible(False) for s in axes[3].spines.values()]
        axes[3].axes.get_xaxis().set_visible(False)
        axes[3].axes.get_yaxis().set_visible(False)

        for i in range(len(hidden_V)):
            ax_num = i+4
            axes[ax_num].plot(timesteps, hidden_V[i], linewidth=2.5)
            axes[ax_num].set_ylim(top=-45)
            for spike_idx in hidden_spikes[i]:
                axes[ax_num].axvline(x=spike_idx, linestyle="solid", color="red", linewidth=3, alpha=0.6)
            axes[ax_num].axes.get_xaxis().set_visible(False)
            axes[ax_num].axes.get_yaxis().set_visible(False)
            [s.set_visible(False) for s in axes[ax_num].spines.values()]

        axes[-1].axes.get_xaxis().set_visible(True)
        axes[-1].set_xlabel("Time [ms]")

        save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
        plt.savefig(save_filename)
        plt.close()