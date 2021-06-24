import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.ssa_input import ssa_input_model, ssa_input_init, SSA_INPUT_PARAMS
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init_classification,
                                           hidden_model, HIDDEN_PARAMS, hidden_init,
                                           feedback_postsyn_model)
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
                                        feedback_wts_model, feedback_wts_init)
import os
from utils import create_poisson_spikes
from math import ceil
from models.parameters import r0, tau_avg_err
# import configparser
# import sys
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from config.plot_config import plot_rcparams

rcParams.update(plot_rcparams)

"""
To run this experiment with a config, you should use 
python xor.py /path/to/config.ini
An example config file is given in config/cfg.ini
If you want to use the default settings (defined in models/parameters.py) 
you can omit the config file and run the script as
python xor.py
"""

# if len(sys.argv) > 1:
#     cfg_file = sys.argv[1]
#     config = configparser.ConfigParser()
#     config.read(cfg_file)
#     experiment_type = str(config["Superspike"]["ExperimentType"])
#     learning_rate = float(config["Superspike"]["LearningRate"])
#     TRIALS = int(config["Superspike"]["Trials"])
#     SUPERSPIKE_PARAMS["r0"] = learning_rate
#
# else:
#     print("Using default arguments")
#     experiment_type = "default"
#     TRIALS = 1001

# TRIALS = 6250
TRIALS = 200

# model_name = experiment_type + "_" + str(learning_rate)
# MODEL_BUILD_DIR = "./"

TIME_FACTOR = 0.1
# UPDATE_TIME = 500  # ms

STIMULUS_TIMESTEPS = 10  # ms
WAIT_TIMESTEPS = 15  # ms
ITI_RANGE = np.arange(50, 60)  # ms
# TEST_ITI = 55  # ms

# N_HIDDEN = 100
N_HIDDEN = 75
N_OUTPUT = 2
WAIT_FREQ = 4  # Hz
STIM_FREQ = 100  # Hz

INPUT_NUM = [['time_ref', 34], ['inp0', 33], ['inp1', 33]]
N_INPUT = sum([i[1] for i in INPUT_NUM])
SPIKE_DT = 0.001  # 1 ms

# SUPERSPIKE_PARAMS["update_t"] = UPDATE_TIME


####### Create poisson spike trains for all input neurons and all trials ###########

"""
`static_spikes_arr` (list) stores the static poisson trains (np.array) for every neuron.
These static spike trains are used for the input presentation based on the selected stimulus.
The length of `static_spikes_arr` is the total number of input neurons `N_INPUT`.
"""

compare_num = STIM_FREQ * (SPIKE_DT * TIME_FACTOR)
static_spikes = np.random.random_sample(size=(N_INPUT, int(STIMULUS_TIMESTEPS / TIME_FACTOR)))
static_spikes = (static_spikes < compare_num).astype(int)
static_spikes = np.transpose(np.nonzero(static_spikes))

static_spikes_arr = []
for i in range(N_INPUT):
    if i in static_spikes[:, 0]:
        neuron_idx = np.where(static_spikes[:, 0] == i)
        neuron_spike_times = static_spikes[neuron_idx, 1]
        # print(neuron_spike_times)
        neuron_spike_times = np.reshape(neuron_spike_times, len(neuron_spike_times[0]))
        neuron_spike_times = np.multiply(neuron_spike_times, TIME_FACTOR)
        static_spikes_arr.append(neuron_spike_times)
    else:
        static_spikes_arr.append(np.array([]))

"""
Every trial consists of three stages: 
stimulus presentation, waiting time, and an intertrial interval (iti).
Here we set up a few more things for the experiment: 
for every trial, we specify the chosen sample and the iti
"""

itis = np.random.choice(ITI_RANGE, size=TRIALS)

SAMPLES = [(0, 0, 0),
           (0, 1, 1),
           (1, 0, 1),
           (1, 1, 0)]

drawn_samples = np.random.choice(np.arange(4), size=TRIALS)

"""
Before we start running the simulation, we will compile all spike times for all neurons
across all trials. This means that the spike trains are not built dynamically but
have to be set before the simulation begins.
`poisson_spikes` will be a list of `N_INPUT` lists: each list is the spike times for each neuron.
"""

poisson_spikes = []

for neuron_idx in range(N_INPUT):

    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)

    for trial_num in range(TRIALS):

        if neuron_idx < INPUT_NUM[0][1]:

            spike_times = np.array(static_spikes_arr[neuron_idx])
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        elif INPUT_NUM[0][1] <= neuron_idx < INPUT_NUM[0][1] + INPUT_NUM[1][1]:

            if SAMPLES[drawn_samples[trial_num]][0] == 1:
                spike_times = np.array(static_spikes_arr[neuron_idx])
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        else:

            if SAMPLES[drawn_samples[trial_num]][1] == 1:
                spike_times = np.array(static_spikes_arr[neuron_idx])
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        time_elapsed += STIMULUS_TIMESTEPS
        wait_plus_iti = WAIT_TIMESTEPS + itis[trial_num]

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, SPIKE_DT, TIME_FACTOR)
        spike_times += time_elapsed
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        time_elapsed += wait_plus_iti

    poisson_spikes.append(neuron_poisson_spikes)

spike_counts = [len(n) for n in poisson_spikes]
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

spikeTimes = np.hstack(poisson_spikes).astype(float)

# """
# Just as we needed to fix the training spike trains for all neurons before
# starting the simulation, we also need to fix the testing spike trains
# for all neurons.
# """
# test_poisson_spikes = []
#
# for neuron_idx in range(N_INPUT):
#
#     time_elapsed = 0
#     neuron_poisson_spikes = np.empty(0)
#
#     for sample_idx in range(len(SAMPLES)):
#
#         if neuron_idx < INPUT_NUM[0][1]:
#
#             spike_times = np.array(static_spikes_arr[neuron_idx])
#             spike_times += time_elapsed
#             neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#         elif INPUT_NUM[0][1] <= neuron_idx < INPUT_NUM[0][1] + INPUT_NUM[1][1]:
#
#             if SAMPLES[sample_idx][0] == 1:
#                 spike_times = np.array(static_spikes_arr[neuron_idx])
#                 spike_times += time_elapsed
#                 neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#         else:
#
#             if SAMPLES[sample_idx][1] == 1:
#                 spike_times = np.array(static_spikes_arr[neuron_idx])
#                 spike_times += time_elapsed
#                 neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#         time_elapsed += STIMULUS_TIMESTEPS
#         wait_plus_iti = WAIT_TIMESTEPS + TEST_ITI
#
#         spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, SPIKE_DT, TIME_FACTOR)
#         spike_times += time_elapsed
#         neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#         time_elapsed += wait_plus_iti
#
#     test_poisson_spikes.append(neuron_poisson_spikes)
#
# test_spike_counts = [len(n) for n in test_poisson_spikes]
# test_end_spike = np.cumsum(test_spike_counts)
# test_start_spike = np.empty_like(test_end_spike)
# test_start_spike[0] = 0
# test_start_spike[1:] = test_end_spike[0:-1]
#
# test_spikeTimes = np.hstack(test_poisson_spikes).astype(float)

# Set up startSpike and endSpike for custom SpikeSourceArray model

ssa_input_init["startSpike"] = start_spike
ssa_input_init["endSpike"] = end_spike

"""
We're ready to build our model!
We define three populations: input, hidden, and output.
We make feedforward connections from input to hidden and from hidden to output.
We also make feedback connections from output to hidden.
"""
model = genn_model.GeNNModel(precision="float", model_name="xor", time_precision="double")
model.dT = 1.0 * TIME_FACTOR

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
inp.set_extra_global_param("spikeTimes", spikeTimes)

hid = model.add_neuron_population("hid", N_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

out = model.add_neuron_population("out", N_OUTPUT, output_model_classification, OUTPUT_PARAMS,
                                  output_init_classification)

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
inp2hid_wt_init = np.random.uniform(low=-a, high=a, size=N_INPUT * N_HIDDEN)
inp2hid.vars["w"].view[:] = inp2hid_wt_init
model.push_var_to_device("inp2hid", "w")
a = 1.0 / np.sqrt(N_HIDDEN)
hid2out_wt_init = np.random.uniform(low=-a, high=a, size=N_HIDDEN * N_OUTPUT)
hid2out.vars["w"].view[:] = hid2out_wt_init
model.push_var_to_device("hid2out", "w")

"""
Before we start the simulation, we need to define some shorthands that
we can use to access different model variables at simulation time. 
"""
out_voltage = out.vars['V'].view
out_window_of_opp = out.vars["window_of_opp"].view
out_S_pred = out.vars['S_pred'].view
out_S_miss = out.vars['S_miss'].view
hid_err_tilda = hid.vars['err_tilda'].view
out_err_tilda = out.vars['err_tilda'].view
inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view
hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view
inp2hid_lambda = inp2hid.vars['lambda'].view
hid2out_lambda = hid2out.vars['lambda'].view
inp2hid_e = inp2hid.vars['e'].view
hid2out_e = hid2out.vars['e'].view
out_err_rise = out.vars["err_rise"].view
out_err_decay = out.vars["err_decay"].view

"""
Here we will use random feedback. The feedback weights need to be set and pushed
to the model only once at the start of the simulation, which we do below.
"""
feedback_wts = np.random.normal(-1.0, 1.0, size=(N_HIDDEN, N_OUTPUT)).flatten()
out2hid.vars['g'].view[:] = feedback_wts
model.push_var_to_device('out2hid', 'g')

IMG_DIR = "imgs_xor"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

"""
Finally, to plot the learning curve, we need to define the variables and data structures as given below.
"""
a = 10.0
b = 5.0
scale_tr_err_flt = 1.0 / ((((a * b) / (a - b)) ** 2) * (a / 2 + b / 2 - 2 * (a * b) / (a + b))) / tau_avg_err
record_avgsqerr = np.empty(0)

time_elapsed = 0  # ms
plot_interval = 5

num_plots = 3
t_start = 0

print("r0 at the start: " + str(r0))

colors = ['royalblue', 'magenta']
c_start = 0

for trial in range(TRIALS):

    if trial % plot_interval == 0:
        itis_plot = []
        targets_plot = []

    if trial % 100 == 0:
        print("\n")
        print("Trial: " + str(trial))

    # Decrease the learning rate every 600th trial
    if trial != 0 and trial % 600 == 0:
        r0 *= 0.1
        inp2hid.vars["r0"].view[:] = r0
        model.push_var_to_device('inp2hid', "r0")
        hid2out.vars["r0"].view[:] = r0
        model.push_var_to_device('hid2out', "r0")
        print("Changed r0 to: " + str(r0))

    # We need some information about this trial...
    target = SAMPLES[drawn_samples[trial]][-1]
    targets_plot.append(target)
    if trial % plot_interval == 0:
        t_start = time_elapsed  # ms
    iti_chosen = itis[trial]  # ms
    itis_plot.append(iti_chosen)
    total_time = STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen  # ms
    # print(total_time)

    # Reinitialize or provide correct values for different variables at the start of the next trial
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
    hid2out_lambda[:] = 0.0
    model.push_var_to_device("hid2out", "lambda")
    inp2hid_lambda[:] = 0.0
    model.push_var_to_device("inp2hid", "lambda")
    hid2out_e[:] = 0.0
    model.push_var_to_device("hid2out", "e")
    inp2hid_e[:] = 0.0
    model.push_var_to_device("inp2hid", "e")
    out_err_tilda[:] = 0.0
    model.push_var_to_device('out', 'err_tilda')
    hid_err_tilda[:] = 0.0
    model.push_var_to_device('hid', 'err_tilda')
    out_err_rise[:] = 0.0
    model.push_var_to_device('out', 'err_rise')
    out_err_decay[:] = 0.0
    model.push_var_to_device('out', 'err_decay')

    # Indicate the correct values for window_of_opp, S_pred, and S_miss before the stimulus is presented
    out_window_of_opp[:] = 1.0
    model.push_var_to_device("out", "window_of_opp")

    S_pred = np.array([1.0, 0.0]) if target == 0 else np.array([0.0, 1.0])
    out.vars['S_pred'].view[:] = S_pred
    model.push_var_to_device("out", "S_pred")

    out_S_miss[:] = 0.0
    model.push_var_to_device("out", "S_miss")

    # To plot the behaviour of the network, we need some data structures to record during the trial
    if trial % plot_interval == 0:
        out0_V = np.empty(0)
        out1_V = np.empty(0)

        # out0_err = np.empty(0)
        # out1_err = np.empty(0)

        inp_spike_ids = np.empty(0)
        inp_spike_times = np.empty(0)

        hid_spike_ids = np.empty(0)
        hid_spike_times = np.empty(0)

    produced_spikes = []

    steps = int(total_time / TIME_FACTOR)
    # print("steps: " + str(steps))

    if trial % plot_interval == 0:
        plot_steps = 0

    # print("plot_steps: ")
    # print(plot_steps)

    for t in range(steps):

        # if this is the timestep immediately after the stimulus time and waiting time
        # where the model is yet to run
        if t == ((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
            # close the window of opportunity
            out_window_of_opp[:] = 0.0
            model.push_var_to_device("out", "window_of_opp")

            # if the model did not produce any spikes
            if len(produced_spikes) == 0:
                # tell the model (the model assigns an error to the appropriate output neuron)
                out_S_miss[:] = 1.0
                model.push_var_to_device("out", "S_miss")

        model.step_time()

        # if this is the timestep immediately after the stimulus time and waiting time
        # where the model has already run
        if t == ((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
            # switch off the missing signal
            out_S_miss[:] = 0.0
            model.push_var_to_device("out", "S_miss")

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

        # At the end of each simulation time step, we would also like to
        # record the variables we need for making a plot
        # if trial % plot_interval == 0:
        model.pull_current_spikes_from_device("inp")
        times = np.ones_like(inp.current_spikes) * model.t
        inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
        inp_spike_times = np.hstack((inp_spike_times, times))

        model.pull_current_spikes_from_device("hid")
        times = np.ones_like(hid.current_spikes) * model.t
        hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
        hid_spike_times = np.hstack((hid_spike_times, times))

        model.pull_var_from_device("out", "V")
        out0_V = np.hstack((out0_V, out.vars["V"].view[0]))
        out1_V = np.hstack((out1_V, out.vars["V"].view[1]))

    time_elapsed += total_time

    plot_steps += steps

    """
	At the end of the trial, we can make a plot
	"""
    if (trial+1) % plot_interval == 0:

        print("Making a plot in trial " + str(trial))

        print("t_start: " + str(t_start))
        print("timee_elapsed: " + str(time_elapsed))
        timesteps_plot = np.linspace(t_start, time_elapsed, num=plot_steps)

        # num_plots = 4

        # fig, axes = plt.subplots(num_plots, sharex=True, figsize=(15, 8))

        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(nrows=3, ncols=1, height_ratios=[2.0, 2.0, 2.0])
        axes = []
        for i in range(3):
            if i == 0:
                axes.append(fig.add_subplot(gs[i]))
            else:
                axes.append(fig.add_subplot(gs[i], sharex=axes[0]))
        plt.subplots_adjust(hspace=0.35)

        # axes[0].plot(timesteps_plot, out0_err, color="royalblue")
        # axes[0].plot(timesteps_plot, out1_err, color="magenta")
        # axes[0].set_ylim(-1.1, 1.1)
        # axes[0].set_title("Error of output neurons")

        axes[0].plot(timesteps_plot, out0_V, color=colors[0])
        axes[0].plot(timesteps_plot, out1_V, color=colors[1])
        axes[0].set_title("Membrane voltage of output neurons")
        axes[0].axhline(y=OUTPUT_PARAMS["Vthresh"])
        # axes[0].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS,
        #                 color="green", linestyle="--")
        # axes[0].axes.get_yaxis().set_visible(False)
        # axes[0].axes.get_xaxis().set_visible(False)
        # [s.set_visible(False) for s in axes[0].spines.values()]

        axes[1].scatter(inp_spike_times, inp_spike_ids)
        axes[1].set_ylim(-1, N_INPUT + 1)
        axes[1].set_title("Input layer spikes")
    #     # axes[1].axhline(y=INPUT_NUM[0][1] - 0.5, color="gray", linestyle="--")
    #     # axes[1].axhline(y=INPUT_NUM[0][1] + INPUT_NUM[1][1] - 0.5, color="gray", linestyle="--")
    #     # axes[1].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS, color="green", linestyle="--")
    #     axes[1].axes.get_xaxis().set_visible(False)
    #     axes[1].axes.get_yaxis().set_visible(False)
    #     [s.set_visible(False) for s in axes[1].spines.values()]
    #
        axes[2].scatter(hid_spike_times, hid_spike_ids)
        axes[2].set_ylim(-1, N_HIDDEN + 1)
        axes[2].set_title("Hidden layer spikes")
    #     # axes[2].axes.get_xaxis().set_visible(False)
    #     axes[2].axes.get_yaxis().set_visible(False)
    #     [s.set_visible(False) for s in axes[2].spines.values()]
    #

        for iti_idx in range(len(itis_plot)):+-
            iti_plot = itis_plot[iti_idx]
            c_end = c_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS
            for ax_idx in range(num_plots):
                axes[ax_idx].axvspan(c_start, c_end, facecolor=colors[targets_plot[iti_idx]], alpha=0.3)
            c_start += STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_plot
    #
    # axes[-1].set_xlabel("Time [ms]")
    # spacing = 100
    # x_ticks_plot = list(range(t_start, time_elapsed, int(ceil(spacing * TIME_FACTOR))))
    # axes[-1].set_xticks(x_ticks_plot)

        save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
        plt.savefig(save_filename)
        plt.close()

"""
With this script, we should see the plots in the `IMG_DIR`.
We can also add some code below to dump all the variables that we have been keeping
track of through the simulation, such as the weights or the best configuration after testing
or the history of errors.
"""
