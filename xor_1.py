import numpy as np
# from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init_classification,
                                           hidden_model, HIDDEN_PARAMS, hidden_init, NUM_HIDDEN)
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init
import os
import random


def create_poisson_spikes(interval, freq, spike_dt):
    compare_num = freq * spike_dt
    spike_train = np.random.random_sample(interval)
    spike_train = (spike_train < compare_num).astype(int)
    spike_times_gen = np.nonzero(spike_train)[0]
    return spike_times_gen

STIMULUS_TIMESTEPS = 10
WAIT_TIMESTEPS = 15
ITI_RANGE = np.arange(15, 20)
TRIALS = 200

STIM_FREQ = 20
WAIT_FREQ = 4

INPUT_NUM = [['time_ref', 34], ['inp0', 33], ['inp1', 33]]
N_INPUT = sum([i[1] for i in INPUT_NUM])
spike_dt = 0.001

####### Create poisson spike trains for all input neurons and all trials ###########

compare_num = STIM_FREQ * spike_dt
static_spikes = np.random.random_sample(size=(N_INPUT, STIMULUS_TIMESTEPS))
static_spikes = (static_spikes < compare_num).astype(int)
static_spikes = np.transpose(np.nonzero(static_spikes))

static_spikes_arr = []
for i in range(N_INPUT):
    if i in static_spikes[:,0]:
        neuron_idx = np.where(static_spikes[:, 0] == i)
        neuron_spike_times = static_spikes[neuron_idx, 1]
        # print(neuron_spike_times)
        neuron_spike_times = np.reshape(neuron_spike_times, len(neuron_spike_times[0]))
        static_spikes_arr.append(neuron_spike_times)
    else:
        static_spikes_arr.append(np.array([]))

# print(static_spikes_arr)

itis = np.random.choice(ITI_RANGE, size=TRIALS)

# print("ITIs chosen: ")
# print(itis)

SAMPLES = [(0, 0, 0),
           (0, 1, 1),
           (1, 0, 1),
           (1, 1, 0)]

drawn_samples = np.random.choice(np.arange(4), size=TRIALS)

# drawn_samples = np.full(shape=TRIALS, fill_value=2)

poisson_spikes = []

freq = WAIT_FREQ

# print("Building time_ref population poisson spikes")

for neuron_idx in range(INPUT_NUM[0][1]):
    # print("Neuron idx: ")
    # print(neuron_idx)
    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)

    for trial_num in range(TRIALS):

        # print("\n")
        # print("Trial num: ")
        # print(trial_num)

        spike_times = np.array(static_spikes_arr[neuron_idx])
        # print(spike_times)
        spike_times += time_elapsed
        # print("Spike times")
        # print(spike_times)
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        # print("Neuron poisson spikes")
        # print(neuron_poisson_spikes)

        time_elapsed += STIMULUS_TIMESTEPS

        wait_plus_iti = WAIT_TIMESTEPS + itis[trial_num]

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, spike_dt)
        spike_times += time_elapsed
        # print("Spike times")
        # print(spike_times)
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        # print("Neuron poisson spikes")
        # print(neuron_poisson_spikes)

        time_elapsed += wait_plus_iti

    poisson_spikes.append(neuron_poisson_spikes)
    # print("Poisson spikes")
    # print(poisson_spikes)

for neuron_idx in range(INPUT_NUM[1][1]):
    # print("Neuron idx: ")
    # print(neuron_idx)
    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)

    for trial_num in range(TRIALS):

        # print("\n")
        # print("Trial num: ")
        # print(trial_num)

        if SAMPLES[drawn_samples[trial_num]][0] == 1:
            spike_times = np.array(static_spikes_arr[neuron_idx + INPUT_NUM[0][1]])
            # print(spike_times)
            spike_times += time_elapsed
            # print("Spike times")
            # print(spike_times)
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
            # print("Neuron poisson spikes")
            # print(neuron_poisson_spikes)

        time_elapsed += STIMULUS_TIMESTEPS

        wait_plus_iti = WAIT_TIMESTEPS + itis[trial_num]

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, spike_dt)
        spike_times += time_elapsed
        # print("Spike times")
        # print(spike_times)
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        # print("Neuron poisson spikes")
        # print(neuron_poisson_spikes)

        time_elapsed += wait_plus_iti

    poisson_spikes.append(neuron_poisson_spikes)
    # print("Poisson spikes")
    # print(poisson_spikes)

for neuron_idx in range(INPUT_NUM[2][1]):
    # print("Neuron idx: ")
    # print(neuron_idx)
    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)

    for trial_num in range(TRIALS):
        # print("\n")
        # print("Trial num: ")
        # print(trial_num)

        if SAMPLES[drawn_samples[trial_num]][1] == 1:
            spike_times = np.array(static_spikes_arr[neuron_idx + INPUT_NUM[0][1]] + INPUT_NUM[1][1])
            # print(spike_times)
            spike_times += time_elapsed
            # print("Spike times")
            # print(spike_times)
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
            # print("Neuron poisson spikes")
            # print(neuron_poisson_spikes)

        time_elapsed += STIMULUS_TIMESTEPS

        wait_plus_iti = WAIT_TIMESTEPS + itis[trial_num]

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, spike_dt)
        spike_times += time_elapsed
        # print("Spike times")
        # print(spike_times)
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        # print("Neuron poisson spikes")
        # print(neuron_poisson_spikes)

        time_elapsed += wait_plus_iti

    poisson_spikes.append(neuron_poisson_spikes)
    # print("Poisson spikes")
    # print(poisson_spikes)

# for p in range(N_INPUT):
#
#     neuron_poisson_spikes = np.empty(0)
#
#     time_elapsed = 0
#
#     for t in range(TRIALS):
#
#         sample_chosen = SAMPLES[0]
#         # print("Sample chosen")
#         # print(sample_chosen)
#         iti_chosen = itis[t]
#
#         interval = WAIT_TIMESTEPS + iti_chosen
#
#         # create spike train for (i) stimulus presentation based on drawn samples
#         # and (ii) inter-trial interval
#
#         if p < INPUT_NUM[0][1]:
#             # time_ref population
#             # always spikes at 8Hz during stimulus presentation
#             spike_times = static_spikes_dict[p]
#             spike_times += time_elapsed
#             neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#             time_elapsed += STIMULUS_TIMESTEPS
#
#             # spikes at 4Hz in inter-trial interval
#             spike_times = create_poisson_spikes(interval, freq, spike_dt)
#             spike_times += time_elapsed
#             neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#             time_elapsed += WAIT_TIMESTEPS + iti_chosen
#
#         # depending on sample chosen, inp0 and inp1 populations may or may not spike during stimulus presentation
#
#         elif INPUT_NUM[0][1] <= p < INPUT_NUM[0][1] + INPUT_NUM[1][1]:
#             # inp0 population
#             if sample_chosen[0] == 1:
#                 spike_times = static_spikes_dict[p]
#                 spike_times += time_elapsed
#                 neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#             time_elapsed += STIMULUS_TIMESTEPS
#
#             # spikes at 4Hz in inter-trial interval
#             spike_times = create_poisson_spikes(interval, freq, spike_dt)
#             spike_times += time_elapsed
#             neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#             time_elapsed += WAIT_TIMESTEPS + iti_chosen
#
#         else:
#             # inp1 population
#             if sample_chosen[1] == 1:
#                 spike_times = static_spikes_dict[p]
#                 spike_times += time_elapsed
#                 neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#             time_elapsed += STIMULUS_TIMESTEPS
#
#             # spikes at 4Hz in inter-trial interval
#             spike_times = create_poisson_spikes(interval, freq, spike_dt)
#             spike_times += time_elapsed
#             neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
#
#             time_elapsed += WAIT_TIMESTEPS + iti_chosen
#
#     poisson_spikes.append(neuron_poisson_spikes)

# for i in range(len(poisson_spikes)):
#     print("Neuron " + str(i))
#     print(poisson_spikes[i])

spike_counts = [len(n) for n in poisson_spikes]
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

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
    threshold_condition_code="$(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]",
    extra_global_params=[("spikeTimes", "scalar*")],
    is_auto_refractory_required=False
)

SSA_INPUT_PARAMS = {"t_rise": 5, "t_decay": 10}

ssa_input_init = {"startSpike": start_spike,
                  "endSpike": end_spike,
                  "z": 0.0,
                  "z_tilda": 0.0}

########### Build model ################
model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = 1.0

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
spikeTimes = np.hstack(poisson_spikes).astype(float)
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

model.build()
model.load()

######### Simulate #############

IMG_DIR = "/home/manvi/Documents/gennzoo/imgs_xor"
out_voltage = out.vars['V'].view
inp_z_tilda = inp.vars["z_tilda"].view
out_window_of_opp = out.vars["window_of_opp"].view
out_S_pred = out.vars['S_pred'].view
out_S_miss = out.vars['S_miss'].view

time_elapsed = 0

for trial in range(TRIALS):

    if trial % 10 == 0:
        print("Trial: " + str(trial))

    out_voltage[:] = OUTPUT_PARAMS["Vrest"]
    model.push_var_to_device('out', "V")

    inp_z_tilda[:] = ssa_input_init["z_tilda"]
    model.push_var_to_device("inp", "z_tilda")

    target = SAMPLES[drawn_samples[trial]][-1]

    t_start = time_elapsed

    produced_spikes = []

    out_window_of_opp[:] = 1.0
    model.push_var_to_device("out", "window_of_opp")

    S_pred = np.array([1.0, 0.0]) if target == 0 else np.array([0.0, 1.0])
    out.vars['S_pred'].view[:] = S_pred
    model.push_var_to_device("out", "S_pred")

    out_S_miss[:] = 0.0
    model.push_var_to_device("out", "S_miss")

    out0_V = np.empty(0)
    out1_V = np.empty(0)
    out0_err = np.empty(0)
    out1_err = np.empty(0)

    inp_spike_ids = np.empty(0)
    inp_spike_times = np.empty(0)

    hid_spike_ids = np.empty(0)
    hid_spike_times = np.empty(0)

    for t in range(STIMULUS_TIMESTEPS):

        model.step_time()

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

        model.pull_var_from_device("out", "V")
        out0_V = np.hstack((out0_V, out.vars["V"].view[0]))
        out1_V = np.hstack((out1_V, out.vars["V"].view[1]))

        model.pull_var_from_device("out", "err_tilda")
        out0_err = np.hstack((out0_err, out.vars["err_tilda"].view[0]))
        out1_err = np.hstack((out1_err, out.vars["err_tilda"].view[1]))

        model.pull_current_spikes_from_device("inp")
        times = np.ones_like(inp.current_spikes) * model.t
        inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
        inp_spike_times = np.hstack((inp_spike_times, times))

        model.pull_current_spikes_from_device("hid")
        times = np.ones_like(hid.current_spikes) * model.t
        hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
        hid_spike_times = np.hstack((hid_spike_times, times))

    time_elapsed += STIMULUS_TIMESTEPS

    for t in range(WAIT_TIMESTEPS):

        model.step_time()

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

        model.pull_var_from_device("out", "V")
        out0_V = np.hstack((out0_V, out.vars["V"].view[0]))
        out1_V = np.hstack((out1_V, out.vars["V"].view[1]))

        model.pull_var_from_device("out", "err_tilda")
        out0_err = np.hstack((out0_err, out.vars["err_tilda"].view[0]))
        out1_err = np.hstack((out1_err, out.vars["err_tilda"].view[1]))

        model.pull_current_spikes_from_device("inp")
        times = np.ones_like(inp.current_spikes) * model.t
        inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
        inp_spike_times = np.hstack((inp_spike_times, times))

        model.pull_current_spikes_from_device("hid")
        times = np.ones_like(hid.current_spikes) * model.t
        hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
        hid_spike_times = np.hstack((hid_spike_times, times))

    out_window_of_opp[:] = 0.0
    model.push_var_to_device("out", "window_of_opp")

    if len(produced_spikes) == 0:
        out_S_miss[:] = 1.0
        model.push_var_to_device("out", "S_miss")

    time_elapsed += WAIT_TIMESTEPS

    iti_chosen = itis[trial]

    for t in range(iti_chosen):

        model.step_time()

        if t == 0:
            out_S_miss[:] = 0.0
            model.push_var_to_device("out", "S_miss")

        model.pull_var_from_device("out", "V")
        out0_V = np.hstack((out0_V, out.vars["V"].view[0]))
        out1_V = np.hstack((out1_V, out.vars["V"].view[1]))

        model.pull_var_from_device("out", "err_tilda")
        out0_err = np.hstack((out0_err, out.vars["err_tilda"].view[0]))
        out1_err = np.hstack((out1_err, out.vars["err_tilda"].view[1]))

        model.pull_current_spikes_from_device("inp")
        times = np.ones_like(inp.current_spikes) * model.t
        inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
        inp_spike_times = np.hstack((inp_spike_times, times))

        model.pull_current_spikes_from_device("hid")
        times = np.ones_like(hid.current_spikes) * model.t
        hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
        hid_spike_times = np.hstack((hid_spike_times, times))

    time_elapsed += iti_chosen

    # print("Creating plot")

    timesteps_plot = list(range(t_start, time_elapsed))

    num_plots = 4

    fig, axes = plt.subplots(num_plots, sharex=True, figsize=(10, 8))

    axes[0].plot(timesteps_plot, out0_err, color="royalblue")
    axes[0].plot(timesteps_plot, out1_err, color="magenta")
    axes[0].set_title("Error of output neurons")

    axes[1].plot(timesteps_plot, out0_V, color="royalblue")
    axes[1].plot(timesteps_plot, out1_V, color="magenta")
    axes[1].set_title("Membrane voltage of output neurons")

    axes[2].scatter(inp_spike_times, inp_spike_ids)
    axes[2].set_title("Input layer spikes")
    axes[2].axhline(y=INPUT_NUM[0][1], color="gray", linestyle="--")
    axes[2].axhline(y=INPUT_NUM[0][1] + INPUT_NUM[1][1], color="gray", linestyle="--")

    axes[3].scatter(hid_spike_times, hid_spike_ids)
    axes[3].set_title("Hidden layer spikes")

    c = 'royalblue' if target == 0 else 'magenta'

    for i in range(num_plots):
        axes[i].axvspan(t_start, t_start + STIMULUS_TIMESTEPS, facecolor=c, alpha=0.3)

    axes[-1].set_xlabel("Time [ms]")

    save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
    plt.savefig(save_filename)
    plt.close()

print("Complete")