import numpy as np
# from scipy.stats import expon
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


def create_poisson_spikes(interval, freq, spike_dt, time_factor):
    """
    Create poisson spike train for 1 neuron
    :param interval: time period to create spikes for (ms)
    :param freq: spiking frequency (Hz)
    :param spike_dt: length of smallest timestep (if 1 ms, use 0.001)
    :return: spike times in ms (np.array) within given time interval for 1 neuron
    """
    compare_num = freq * (spike_dt * TIME_FACTOR)
    spike_train = np.random.random_sample(int(interval / time_factor))
    spike_train = (spike_train < compare_num).astype(int)
    spike_times_gen = np.nonzero(spike_train)[0]
    spike_times_gen = np.multiply(spike_times_gen, time_factor)
    return spike_times_gen


wmax = SUPERSPIKE_PARAMS['wmax']
lr = str(SUPERSPIKE_PARAMS["r0"])[2:]
model_name = lr + "_" + str(int(wmax))
# IMG_DIR = "/data/p286814/runs_3007/imgs_xor_" + model_name
IMG_DIR = "/home/manvi/Documents/gennzoo/imgs_xor_test"

MODEL_BUILD_DIR = os.environ.get('TMPDIR')

TIME_FACTOR = 0.1

# following timesteps are given in milliseconds
STIMULUS_TIMESTEPS = 10
WAIT_TIMESTEPS = 15
ITI_RANGE = np.arange(50, 60)
TEST_ITI = 55

TRIALS = 100
NUM_HIDDEN = 100
WAIT_FREQ = 4  # Hz
STIM_FREQ = 100  # Hz

INPUT_NUM = [['time_ref', 34], ['inp0', 33], ['inp1', 33]]
N_INPUT = sum([i[1] for i in INPUT_NUM])
spike_dt = 0.001  # 1 ms

# dump dummy object to test if script is able to access and dump things into directory
test_obj = {'a': 'test',
            'b': 'test'}

with open(os.path.join(IMG_DIR, "test.pkl"), 'wb') as f:
    pkl.dump(test_obj, f)

####### Create poisson spike trains for all input neurons and all trials ###########

# static_spikes_arr (list) stores the static poisson trains (np.array) for every neuron
# these static spike trains are used for the input presentation based on the selected stimulus
# the length of static_spikes_arr is the total number of input neurons
compare_num = STIM_FREQ * (spike_dt * TIME_FACTOR)
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

# Every trial consists of three stages: stimulus presentation, waiting time, and intertrial interval (iti)
# Here we set up a few more things for the experiment: chosen sample and iti for every trial

itis = np.random.choice(ITI_RANGE, size=TRIALS)

SAMPLES = [(0, 0, 0),
           (0, 1, 1),
           (1, 0, 1),
           (1, 1, 0)]

drawn_samples = np.random.choice(np.arange(4), size=TRIALS)

##### Build poisson spikes for all neurons ######

# poisson_spikes is a list of N_INPUT lists: each list is the spike times for each neuron
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

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, spike_dt, TIME_FACTOR)
        spike_times += time_elapsed
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        time_elapsed += wait_plus_iti

    poisson_spikes.append(neuron_poisson_spikes)

# print("Poisson spikes: ")
# for i in range(N_INPUT):
#     print("Neuron idx: " + str(i))
#     print(poisson_spikes[i])

spike_counts = [len(n) for n in poisson_spikes]
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

spikeTimes = np.hstack(poisson_spikes).astype(float)

##### Build test poisson spikes for all neurons ######

# poisson_spikes is a list of N_INPUT lists: each list is the spike times for each neuron
test_poisson_spikes = []

for neuron_idx in range(N_INPUT):

    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)

    for sample_idx in range(len(SAMPLES)):

        if neuron_idx < INPUT_NUM[0][1]:

            spike_times = np.array(static_spikes_arr[neuron_idx])
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        elif INPUT_NUM[0][1] <= neuron_idx < INPUT_NUM[0][1] + INPUT_NUM[1][1]:

            if SAMPLES[sample_idx][0] == 1:
                spike_times = np.array(static_spikes_arr[neuron_idx])
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        else:

            if SAMPLES[sample_idx][1] == 1:
                spike_times = np.array(static_spikes_arr[neuron_idx])
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        time_elapsed += STIMULUS_TIMESTEPS
        wait_plus_iti = WAIT_TIMESTEPS + TEST_ITI

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, spike_dt, TIME_FACTOR)
        spike_times += time_elapsed
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        time_elapsed += wait_plus_iti

    test_poisson_spikes.append(neuron_poisson_spikes)

test_spike_counts = [len(n) for n in test_poisson_spikes]
test_end_spike = np.cumsum(test_spike_counts)
test_start_spike = np.empty_like(test_end_spike)
test_start_spike[0] = 0
test_start_spike[1:] = test_end_spike[0:-1]

test_spikeTimes = np.hstack(test_poisson_spikes).astype(float)

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

ssa_input_init = {"startSpike": start_spike,
                  "endSpike": end_spike,
                  "z": 0.0,
                  "z_tilda": 0.0}

test_ssa_input_init = {"startSpike": test_start_spike,
                       "endSpike": test_end_spike}

test_lif_init = {"V": -60.0,
                 "RefracTime": 0.0}

TEST_LIF_PARAMS = {"C": 10.0,
                   "TauM": 10.0,
                   "Vrest": -60.0,
                   "Vreset": -60.0,
                   "Vthresh": -50.0,
                   "Ioffset": 0.0,
                   "TauRefrac": 5.0}

########### Build model ################
model = genn_model.GeNNModel("float", model_name)
model.dT = 1.0 * TIME_FACTOR

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
inp.set_extra_global_param("spikeTimes", spikeTimes)

hid = model.add_neuron_population("hid", NUM_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

out = model.add_neuron_population("out", 2, output_model_classification, OUTPUT_PARAMS, output_init_classification)

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

# model.build(path_to_model=MODEL_BUILD_DIR)
# model.load(path_to_model=MODEL_BUILD_DIR + "/")

model.build()
model.load()

# print("Built main model")

######### Training #############

# Access variables during training time
out_voltage = out.vars['V'].view
out_window_of_opp = out.vars["window_of_opp"].view
out_S_pred = out.vars['S_pred'].view
out_S_miss = out.vars['S_miss'].view
inp2hid_trial_length = inp2hid.vars["trial_length"].view
hid2out_trial_length = hid2out.vars["trial_length"].view
inp2hid_trial_end_t = inp2hid.vars["trial_end_t"].view
hid2out_trial_end_t = hid2out.vars["trial_end_t"].view
hid_err_tilda = hid.vars['err_tilda'].view
out_err_tilda = out.vars['err_tilda'].view

inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view
hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view

out2hid_wts = out2hid.vars['g'].view

# Data structures for recording weights at the end of every trial
wts_inp2hid = np.array([np.empty(0) for _ in range(N_INPUT * NUM_HIDDEN)])
wts_hid2out = np.array([np.empty(0) for _ in range(NUM_HIDDEN * 2)])

time_elapsed = 0

# Incorporate this into the model -- should go in the weight update model
# Random feedback
# feedback_wts = np.random.normal(0.0, 1.0, size=(NUM_HIDDEN, 2))

# Record best network config encountered so far
best_wts = {'inp2hid': 0,
            'hid2out': 0}
best_err = np.inf
best_acc = 0
best_trial = 0

plot_interval = 1

for trial in range(TRIALS):

    if trial % 1 == 0:
        print("Trial: " + str(trial))

    # Important to record for this trial
    target = SAMPLES[drawn_samples[trial]][-1]
    t_start = time_elapsed  # ms
    iti_chosen = itis[trial]  # ms
    total_time = STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen  # ms

    # print("Start time: " + str(t_start) + " ms")
    # print("End time: " + str(t_start + total_time) + " ms")

    # Symmetric feedback
    model.pull_var_from_device("hid2out", "w")
    h2o_weights = hid2out.get_var_values("w")
    # feedback_wts = np.reshape(h2o_weights, newshape=(NUM_HIDDEN, 2))
    out2hid_wts[:] = h2o_weights
    model.push_var_to_device("out2hid", "g")

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

    out_err_tilda[:] = 0.0
    model.push_var_to_device('out', 'err_tilda')

    hid_err_tilda[:] = 0.0
    model.push_var_to_device('hid', 'err_tilda')

    inp2hid_trial_length[:] = float(STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen)
    model.push_var_to_device("inp2hid", "trial_length")

    hid2out_trial_length[:] = float(STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen)
    model.push_var_to_device("hid2out", "trial_length")

    inp2hid_trial_end_t[:] = float(time_elapsed + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen - 1)
    model.push_var_to_device("inp2hid", "trial_end_t")

    hid2out_trial_end_t[:] = float(time_elapsed + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen - 1)
    model.push_var_to_device("hid2out", "trial_end_t")

    out.vars["err_rise"].view[:] = 0.0
    model.push_var_to_device('out', 'err_rise')
    out.vars["err_decay"].view[:] = 0.0
    model.push_var_to_device('out', 'err_decay')

    # Indicate the correct values for window_of_opp, S_pred, and S_miss before the stimulus is presented
    out_window_of_opp[:] = 1.0
    model.push_var_to_device("out", "window_of_opp")

    S_pred = np.array([1.0, 0.0]) if target == 0 else np.array([0.0, 1.0])
    out.vars['S_pred'].view[:] = S_pred
    model.push_var_to_device("out", "S_pred")

    out_S_miss[:] = 0.0
    model.push_var_to_device("out", "S_miss")

    # Initialize some data structures for recording various things during the trial
    if trial % plot_interval == 0:
        out0_V = np.empty(0)
        out1_V = np.empty(0)
        out0_err = np.empty(0)
        out1_err = np.empty(0)

        inp_spike_ids = np.empty(0)
        inp_spike_times = np.empty(0)

        hid_spike_ids = np.empty(0)
        hid_spike_times = np.empty(0)

        hid_err_tilda_arr = [np.empty(0) for _ in range(NUM_HIDDEN)]

    produced_spikes = []
    err_sum = 0

    steps = int(total_time / TIME_FACTOR)

    for t in range(steps):

        # model.pull_var_from_device("out", "err_tilda")
        # err_output = out.vars["err_tilda"].view[:]
        # err_hidden = np.sum(np.multiply(feedback_wts, err_output), axis=1)
        # hid_err_tilda[:] = err_hidden
        # model.push_var_to_device('hid', 'err_tilda')

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

        err_sum += np.sum(np.abs(out.vars["err_tilda"].view[:]))

        if trial % plot_interval == 0:
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

            model.pull_var_from_device("out", "err_tilda")
            out0_err = np.hstack((out0_err, out.vars["err_tilda"].view[0]))
            out1_err = np.hstack((out1_err, out.vars["err_tilda"].view[1]))

            model.pull_var_from_device("hid", "err_tilda")
            hid_err_tilda_new = hid_err_tilda[:]
            for i in range(NUM_HIDDEN):
                hid_err_tilda_arr[i] = np.hstack((hid_err_tilda_arr[i], hid_err_tilda_new[i]))

    time_elapsed += total_time

    # Record the weights at the end of the trial
    model.pull_var_from_device("inp2hid", "w")
    weights = inp2hid.get_var_values("w")
    weights = np.reshape(weights, (weights.shape[0], 1))
    wts_inp2hid = np.concatenate((wts_inp2hid, weights), axis=1)

    model.pull_var_from_device("hid2out", "w")
    h2o_weights = hid2out.get_var_values("w")
    weights = np.reshape(h2o_weights, (h2o_weights.shape[0], 1))
    wts_hid2out = np.concatenate((wts_hid2out, weights), axis=1)

    # # ############ TESTING ###########
    #
    # if err_sum <= best_err:
    #
    #     print("Testing")
    #     print("For trial: " + str(trial))
    #
    #     best_err = err_sum
    #
    #     model.pull_var_from_device("inp2hid", "w")
    #     i2h_weights = inp2hid.get_var_values("w")
    #     model.pull_var_from_device("hid2out", "w")
    #     h2o_weights = hid2out.get_var_values("w")
    #
    #     test_network = genn_model.GeNNModel("float", "test_network" + model_name)
    #     test_network.dT = 1.0 * TIME_FACTOR
    #
    #     test_inp = test_network.add_neuron_population("inp", N_INPUT, "SpikeSourceArray", {},
    #                                                   test_ssa_input_init)
    #     test_inp.set_extra_global_param("spikeTimes", test_spikeTimes)
    #
    #     test_hid = test_network.add_neuron_population("hid", NUM_HIDDEN, "LIF", TEST_LIF_PARAMS, test_lif_init)
    #
    #     test_out = test_network.add_neuron_population("out", 2, "LIF", TEST_LIF_PARAMS,
    #                                                   test_lif_init)
    #
    #     test_inp2hid = test_network.add_synapse_population("inp2hid", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    #                                                        test_inp, test_hid,
    #                                                        "StaticPulse", {}, {"g": i2h_weights.flatten()}, {}, {},
    #                                                        "ExpCurr", {"tau": 5.0}, {})
    #
    #     test_hid2out = test_network.add_synapse_population("hid2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    #                                                        test_hid, test_out,
    #                                                        "StaticPulse", {}, {"g": h2o_weights.flatten()}, {}, {},
    #                                                        "ExpCurr", {"tau": 5.0}, {})
    #
    #     test_network.build(path_to_model=MODEL_BUILD_DIR)
    #     test_network.load(path_to_model=MODEL_BUILD_DIR + "/")
    #
    #     # test_network.build()
    #     # test_network.load()
    #
    #     # print("Finished building test network.")
    #
    #     num_correct = 0
    #
    #     # test_time_elapsed = 0
    #
    #     for sample_idx in range(len(SAMPLES)):
    #
    #         # test_inp_spike_ids = np.empty(0)
    #         # test_inp_spike_times = np.empty(0)
    #         #
    #         # test_hid_spike_ids = np.empty(0)
    #         # test_hid_spike_times = np.empty(0)
    #         #
    #         # test_t_start = test_time_elapsed
    #
    #         test_target = SAMPLES[sample_idx][-1]
    #         test_non_target = 1 - test_target
    #
    #         test_target_spikes = []
    #         test_non_target_spikes = []
    #
    #         test_steps = int((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + TEST_ITI) / TIME_FACTOR)
    #
    #         for t in range(test_steps):
    #
    #             test_network.step_time()
    #
    #             # test_network.pull_current_spikes_from_device("inp")
    #             # times = np.ones_like(test_inp.current_spikes) * test_network.t
    #             # test_inp_spike_ids = np.hstack((test_inp_spike_ids, test_inp.current_spikes))
    #             # test_inp_spike_times = np.hstack((test_inp_spike_times, times))
    #             #
    #             # test_network.pull_current_spikes_from_device("hid")
    #             # times = np.ones_like(test_hid.current_spikes) * test_network.t
    #             # test_hid_spike_ids = np.hstack((test_hid_spike_ids, test_hid.current_spikes))
    #             # test_hid_spike_times = np.hstack((test_hid_spike_times, times))
    #
    #             if t < int((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
    #                 test_network.pull_current_spikes_from_device("out")
    #                 if test_target in test_out.current_spikes:
    #                     test_target_spikes.append(test_network.t)
    #                 if test_non_target in test_out.current_spikes:
    #                     test_non_target_spikes.append(test_network.t)
    #
    #         if len(test_target_spikes) != 0 and len(test_non_target_spikes) == 0:
    #             num_correct += 1
    #
    #         # test_time_elapsed += STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + TEST_ITI
    #
    #         # # PLOTTING FOR TESITNG
    #         #
    #         # num_plots = 2
    #         #
    #         # fig, axes = plt.subplots(num_plots, sharex=True, figsize=(15, 8))
    #         #
    #         # axes[0].scatter(test_inp_spike_times, test_inp_spike_ids)
    #         # axes[0].set_ylim(-1, N_INPUT + 1)
    #         # axes[0].set_title("Input layer spikes")
    #         # axes[0].axhline(y=INPUT_NUM[0][1] - 0.5, color="gray", linestyle="--")
    #         # axes[0].axhline(y=INPUT_NUM[0][1] + INPUT_NUM[1][1] - 0.5, color="gray", linestyle="--")
    #         # # axes[0].axvline(x=t_start + STIMULUS_TIMESTEPS, color="green", linestyle="--")
    #         # axes[0].axvline(x=test_t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS, color="green", linestyle="--")
    #         #
    #         # axes[1].scatter(test_hid_spike_times, test_hid_spike_ids)
    #         # axes[1].set_ylim(-1, NUM_HIDDEN + 1)
    #         # axes[1].set_title("Hidden layer spikes")
    #         #
    #         # c = 'royalblue' if test_target == 0 else 'magenta'
    #         #
    #         # for i in range(num_plots):
    #         #     axes[i].axvspan(test_t_start, test_t_start + STIMULUS_TIMESTEPS, facecolor=c, alpha=0.3)
    #         #
    #         # axes[-1].set_xlabel("Time [ms] x 0.1")
    #         # # axes.set_xlabel("Time [ms]")
    #         # x_ticks_plot = list(range(test_t_start, test_time_elapsed, int(ceil(5 * TIME_FACTOR))))
    #         # axes[-1].set_xticks(x_ticks_plot)
    #         #
    #         # save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + "_test" + str(sample_idx) + ".png")
    #         # plt.savefig(save_filename)
    #         # plt.close()
    #
    #     accuracy = num_correct / 4
    #
    #     if accuracy > best_acc:
    #         test_network.pull_var_from_device("inp2hid", "g")
    #         best_wts['inp2hid'] = test_inp2hid.get_var_values("g")
    #         test_network.pull_var_from_device("hid2out", "g")
    #         best_wts['hid2out'] = test_hid2out.get_var_values("g")
    #         best_acc = accuracy
    #         best_trial = trial

    ########## Make plots similar to Fig. 5b from the paper #############

    # if (3000 <= trial < 6000 and trial % 10 == 0) or (trial >= 6000):
    if trial % plot_interval == 0:

        # print("Input spike times: ")
        # print(inp_spike_times)
        # print("Input spike ids: ")
        # print(inp_spike_ids)

        timesteps_plot = np.linspace(t_start, time_elapsed, num=steps)

        num_plots = 5

        fig, axes = plt.subplots(num_plots, sharex=True, figsize=(15, 8))

        axes[0].plot(timesteps_plot, out0_err, color="royalblue")
        axes[0].plot(timesteps_plot, out1_err, color="magenta")
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].set_title("Error of output neurons")

        axes[1].plot(timesteps_plot, out0_V, color="royalblue")
        axes[1].plot(timesteps_plot, out1_V, color="magenta")
        axes[1].set_title("Membrane voltage of output neurons")
        axes[1].axhline(y=OUTPUT_PARAMS["Vthresh"])
        axes[1].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS,
                        color="green", linestyle="--")

        axes[2].scatter(inp_spike_times, inp_spike_ids)
        axes[2].set_ylim(-1, N_INPUT + 1)
        axes[2].set_title("Input layer spikes")
        axes[2].axhline(y=INPUT_NUM[0][1] - 0.5, color="gray", linestyle="--")
        axes[2].axhline(y=INPUT_NUM[0][1] + INPUT_NUM[1][1] - 0.5, color="gray", linestyle="--")
        # axes[2].axvline(x=t_start + STIMULUS_TIMESTEPS, color="green", linestyle="--")
        axes[2].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS, color="green", linestyle="--")

        axes[3].scatter(hid_spike_times, hid_spike_ids)
        axes[3].set_ylim(-1, NUM_HIDDEN + 1)
        axes[3].set_title("Hidden layer spikes")

        for i in range(NUM_HIDDEN):
            axes[4].plot(timesteps_plot, hid_err_tilda_arr[i])
        axes[4].set_title("Hidden err_tilda")

        c = 'royalblue' if target == 0 else 'magenta'

        for i in range(num_plots):
            axes[i].axvspan(t_start, t_start + STIMULUS_TIMESTEPS, facecolor=c, alpha=0.3)

        axes[-1].set_xlabel("Time [ms] x 0.1")
        # axes.set_xlabel("Time [ms]")
        x_ticks_plot = list(range(t_start, time_elapsed, int(ceil(5 * TIME_FACTOR))))
        axes[-1].set_xticks(x_ticks_plot)

        save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
        plt.savefig(save_filename)
        plt.close()

# Plot weights as a pixel plot with a colorbar
print("Creating wts_inp2hid")
plt.figure(figsize=(20, 50))
plt.imshow(wts_inp2hid, cmap='gray')
plt.colorbar()
plt.yticks(list(range(wts_inp2hid.shape[0])))
plt.xticks(list(range(wts_inp2hid.shape[1])))
save_filename = os.path.join(IMG_DIR, "wts_inp2hid.png")
plt.savefig(save_filename)
plt.close()
print("Creating wts_hid2out")
plt.figure()
plt.imshow(wts_hid2out, cmap='gray')
plt.colorbar()
plt.yticks(list(range(wts_hid2out.shape[0])))
plt.xticks(list(range(wts_hid2out.shape[1])))
save_filename = os.path.join(IMG_DIR, "wts_hid2out.png")
plt.savefig(save_filename)
plt.close()

pkl_dict = {'wmax': SUPERSPIKE_PARAMS['wmax'],
            'wmin': SUPERSPIKE_PARAMS['wmin'],
            'trials': TRIALS,
            'hidden_num': NUM_HIDDEN,
            'learning_rate': SUPERSPIKE_PARAMS['r0'],
            'best_trial': best_trial,
            'feedback': 'symmetric',
            'best_acc': best_acc,
            'inp2hid': best_wts['inp2hid'],
            'hid2out': best_wts['hid2out']}

filename = os.path.join(IMG_DIR, 'config.pkl')

with open(os.path.join(IMG_DIR, "config.pkl"), 'wb') as fi:
    pkl.dump(pkl_dict, fi)

print("Complete.")
