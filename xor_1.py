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
ITI_RANGE = np.arange(5, 10)
TRIALS = 2

STIM_FREQ = 8
WAIT_FREQ = 4

INPUT_NUM = [['time_ref', 34], ['inp0', 33], ['inp1', 33]]

####### Create poisson spike trains for all input neurons and all trials ###########

itis = np.random.choice(ITI_RANGE, size=TRIALS)

SAMPLES = [(0, 0, 0),
           (0, 1, 1),
           (1, 0, 1),
           (1, 1, 0)]

drawn_samples = np.random.choice(np.arange(4), size=TRIALS)

spike_dt = 0.001
N_INPUT = sum([i[1] for i in INPUT_NUM])
poisson_spikes = []

for p in range(N_INPUT):

    neuron_poisson_spikes = np.empty(0)

    time_elapsed = 0

    for t in range(TRIALS):

        sample_chosen = SAMPLES[drawn_samples[t]]
        iti_chosen = itis[t]

        # create spike train for (i) stimulus presentation based on drawn samples
        # and (ii) inter-trial interval

        if p < INPUT_NUM[0][1]:
            # time_ref population
            # always spikes at 8Hz during stimulus presentation
            freq = STIM_FREQ
            interval = STIMULUS_TIMESTEPS
            spike_times = create_poisson_spikes(interval, freq, spike_dt)
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

            time_elapsed += STIMULUS_TIMESTEPS

            # spikes at 4Hz in inter-trial interval
            freq = WAIT_FREQ
            interval = WAIT_TIMESTEPS + iti_chosen
            spike_times = create_poisson_spikes(interval, freq, spike_dt)
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

            time_elapsed += WAIT_TIMESTEPS + iti_chosen

        # depending on sample chosen, inp0 and inp1 populations may or may not spike during stimulus presentation

        elif INPUT_NUM[0][1] <= p < INPUT_NUM[0][1] + INPUT_NUM[1][1]:
            # inp0 population
            if sample_chosen[0] == 1:
                freq = STIM_FREQ
                interval = STIMULUS_TIMESTEPS
                spike_times = create_poisson_spikes(interval, freq, spike_dt)
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

            time_elapsed += STIMULUS_TIMESTEPS

            # spikes at 4Hz in inter-trial interval
            freq = WAIT_FREQ
            interval = WAIT_TIMESTEPS + iti_chosen
            spike_times = create_poisson_spikes(interval, freq, spike_dt)
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

            time_elapsed += WAIT_TIMESTEPS + iti_chosen

        else:
            # inp1 population
            if sample_chosen[1] == 1:
                freq = STIM_FREQ
                interval = STIMULUS_TIMESTEPS
                spike_times = create_poisson_spikes(interval, freq, spike_dt)
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

            time_elapsed += STIMULUS_TIMESTEPS

            # spikes at 4Hz in inter-trial interval
            freq = WAIT_FREQ
            interval = WAIT_TIMESTEPS + iti_chosen
            spike_times = create_poisson_spikes(interval, freq, spike_dt)
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

            time_elapsed += WAIT_TIMESTEPS + iti_chosen

    poisson_spikes.append(neuron_poisson_spikes)

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

    for t in range(STIMULUS_TIMESTEPS):

        model.step_time()

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

    time_elapsed += STIMULUS_TIMESTEPS

    for t in range(WAIT_TIMESTEPS):

        model.step_time()

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

    time_elapsed += iti_chosen

print("Complete")