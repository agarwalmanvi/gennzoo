import numpy as np
# from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init,
                                           hidden_model, HIDDEN_PARAMS, hidden_init, NUM_HIDDEN)
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init
import os

STIMULUS_TIMESTEPS = 10
WAIT_TIMESTEPS = 15
ITI_RANGE = list(range(5,10))
TRIALS = 2000

INPUT_NUM = [('time_ref', 34), ('inp0', 33), ('inp1', 33)]

###### Helper functions #######

def create_poisson_spikes(interval, freq, spike_dt, n_input):
    poisson_spike_arr = []
    compare_num = freq * spike_dt
    for p in range(n_input):
        spike_train = np.random.random_sample(interval)
        spike_train = (spike_train < compare_num).astype(int)
        poisson_spike_arr.append(np.nonzero(spike_train)[0])

    # poisson_spike_arr is a list of 100 lists: each list is the spike times for each neuron

    # Count spikes each neuron should emit
    spike_counts = [len(n) for n in poisson_spike_arr]

    # spike_counts is a list of 100 elements, each element corresponding to the number of spikes each neuron emits

    # Get start and end indices of each spike sources section
    end_spike_arr = np.cumsum(spike_counts)
    start_spike_arr = np.empty_like(end_spike_arr)
    start_spike_arr[0] = 0
    start_spike_arr[1:] = end_spike_arr[0:-1]

    return poisson_spike_arr, start_spike_arr, end_spike_arr


######### Create poisson spike trains for static stimulus ############

end_spikes = []
start_spikes = []
poisson_spikes = []

for pop in INPUT_NUM:

    poisson_spike, start_spike, end_spike = create_poisson_spikes(interval = STIMULUS_TIMESTEPS,
                                                                  freq=8, spike_dt = 0.001,
                                                                  n_input = pop[1])

    poisson_spikes.append(poisson_spike)
    start_spikes.append(start_spike)
    end_spikes.append(end_spike)

######### Define XOR patterns ##############

SAMPLES = [(0, 0, 0),
           (0, 1, 1),
           (1, 0, 1),
           (1, 1, 0)]

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

# define init when creating the three input populations

########### Build model ################
model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = 1.0

input_pops = {}
spikeTimes_arr = []

for idx in range(len(INPUT_NUM)):
    pop_name = INPUT_NUM[idx][0]
    pop_num = INPUT_NUM[idx][1]

    ssa_input_init = {"z": 0.0,
                      "z_tilda": 0.0,
                      "startSpike": start_spikes[idx],
                      "endSpike": end_spikes[idx]}

    input_pops[pop_name] = model.add_neuron_population(pop_name, pop_num,
                                                       ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)

    spikeTimes = np.hstack(poisson_spikes[idx]).astype(float)
    input_pops[pop_name].set_extra_global_param("spikeTimes", spikeTimes)
    spikeTimes_arr.append(spikeTimes)

hid = model.add_neuron_population("hid", NUM_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

out = model.add_neuron_population("out", 2, output_model_classification, OUTPUT_PARAMS, output_init)

inp2hid = {}

for pop in input_pops:
    inp2hid[pop] = model.add_synapse_population(pop + str("2hid"), "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                                input_pops[pop], hid,
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

time_elapsed = 0

for trial in range(TRIALS):

    print("Trial: " + str(trial))

    iti = np.random.choice(ITI_RANGE)
    sample = np.random.choice(SAMPLES)

    inp0_switch = sample[0]
    inp1_switch = sample[1]
    target = sample[2]

    produced_spikes = []

    # reset variables before presenting stimulus
    out_voltage[:] = OUTPUT_PARAMS["Vrest"]
    model.push_var_to_device('out', "V")

    for pop in input_pops:
        input_pops[pop].vars["z_tilda"].view[:] = 0.0
        model.push_var_to_device(pop, 'z_tilda')

    # set firing patterns based on sample
    for idx in range(len(INPUT_NUM)):

        pop_name = INPUT_NUM[idx][0]

        if pop_name == "inp0" and inp0_switch == 1:

            input_pops[pop_name].vars["startSpike"].view[:] = start_spikes[idx]
            input_pops[pop_name].vars["endSpike"].view[:] = end_spikes[idx]

            spikeTimes_chosen = spikeTimes_arr[idx]

            spikeTimes_chosen += time_elapsed
            input_pops[pop_name].extra_global_params['spikeTimes'].view[:] = spikeTimes_chosen
            model.push_extra_global_param_to_device(pop_name, "spikeTimes")

        if pop_name == "inp1" and inp1_switch == 1:

            input_pops[pop_name].vars["startSpike"].view[:] = start_spikes[idx]
            input_pops[pop_name].vars["endSpike"].view[:] = end_spikes[idx]

            spikeTimes_chosen = spikeTimes_arr[idx]

            spikeTimes_chosen += time_elapsed
            input_pops[pop_name].extra_global_params['spikeTimes'].view[:] = spikeTimes_chosen
            model.push_extra_global_param_to_device(pop_name, "spikeTimes")

        if pop_name == "time_ref":

            input_pops[pop_name].vars["startSpike"].view[:] = start_spikes[idx]
            input_pops[pop_name].vars["endSpike"].view[:] = end_spikes[idx]

            spikeTimes_chosen = spikeTimes_arr[idx]

            spikeTimes_chosen += time_elapsed
            input_pops[pop_name].extra_global_params['spikeTimes'].view[:] = spikeTimes_chosen
            model.push_extra_global_param_to_device(pop_name, "spikeTimes")

    # set window_of_opp to 1
    out.extra_global_params['window_of_opp'].view[:] = 1.0
    model.push_extra_global_param_to_device("out", "window_of_opp")

    # set identity of target neuron
    S_pred = np.array([1.0, 0.0]) if target == 0 else np.array([0.0, 1.0])
    out.extra_global_params['S_pred'].view[:] = S_pred
    model.push_extra_global_param_to_device("out", "S_pred")

    # set S_miss to 0
    out.extra_global_params['S_miss'].view[:] = 0.0
    model.push_extra_global_param_to_device("out", "S_miss")

    for t in range(STIMULUS_TIMESTEPS):

        model.step_time()

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

    time_elapsed += STIMULUS_TIMESTEPS

    # Create spikes trains for time between stimulus presentations

    wait_plus_iti = int(WAIT_TIMESTEPS + iti)

    for idx in range(len(INPUT_NUM)):
        pop_name = INPUT_NUM[idx][0]
        pop_num = INPUT_NUM[idx][1]

        poisson_spike, start_spike, end_spike = create_poisson_spikes(interval=wait_plus_iti,
                                                                      freq=4, spike_dt=0.001,
                                                                      n_input=pop_num)

        temp_spikeTimes = np.hstack(poisson_spike).astype(float)
        temp_spikeTimes += time_elapsed
        input_pops[pop_name].extra_global_params['spikeTimes'].view[:] = temp_spikeTimes
        model.push_extra_global_param_to_device(pop_name, "spikeTimes")

        input_pops[pop_name].vars["startSpike"].view[:] = start_spike
        input_pops[pop_name].vars["endSpike"].view[:] = end_spike

    for t in range(wait_plus_iti):

        if t == WAIT_TIMESTEPS:
            # set window_of_opp to 0 at the end of wait timesteps
            out.extra_global_params['window_of_opp'].view[:] = 0.0
            model.push_extra_global_param_to_device("out", "window_of_opp")

            if len(produced_spikes) == 0:
                out.extra_global_params['S_miss'].view[:] = 1.0
                model.push_extra_global_param_to_device("out", "S_miss")

        if t == WAIT_TIMESTEPS + 1:
            out.extra_global_params['S_miss'].view[:] = 0.0
            model.push_extra_global_param_to_device("out", "S_miss")

        model.step_time()

    time_elapsed += wait_plus_iti

    print("Creating plot")






