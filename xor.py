import numpy as np
# from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init,
                                           hidden_model, HIDDEN_PARAMS, hidden_init, NUM_HIDDEN)
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init
import os

STIMULUS_TIMESTEPS = 10.0
WAIT_TIMESTEPS = 15.0
ITI_RANGE = np.arange(5.0, 10.0)
TRIALS = 2000

INPUT_NUM = [('time_ref', 34), ('inp0', 33), ('inp1', 33)]

######### Create three poisson spike trains ############

end_spikes = []
start_spikes = []
poisson_spikes = []

for pop in INPUT_NUM:
    poisson_spike = []
    interval = STIMULUS_TIMESTEPS
    freq = 8
    spike_dt = 0.001
    N_INPUT = pop[1]
    compare_num = freq * spike_dt
    for p in range(N_INPUT):
        spike_train = np.random.random_sample(interval)
        spike_train = (spike_train < compare_num).astype(int)
        poisson_spike.append(np.nonzero(spike_train)[0])

    # poisson_spike is a list of 100 lists: each list is the spike times for each neuron

    # Count spikes each neuron should emit
    spike_counts = [len(n) for n in poisson_spike]

    # spike_counts is a list of 100 elements, each element corresponding to the number of spikes each neuron emits

    # Get start and end indices of each spike sources section
    end_spike = np.cumsum(spike_counts)
    start_spike = np.empty_like(end_spike)
    start_spike[0] = 0
    start_spike[1:] = end_spike[0:-1]

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

for trial in range(TRIALS):

    print("Trial: " + str(trial))

    iti = np.random.choice(ITI_RANGE)
    sample = np.random.choice(SAMPLES)

    inp0_switch = sample[0]
    inp1_switch = sample[1]
    target = sample[2]

    produced_spikes = []

    for t in range(STIMULUS_TIMESTEPS):

        if t == 0:

            out_voltage[:] = OUTPUT_PARAMS["Vrest"]
            model.push_var_to_device('out', "V")

            for pop in input_pops:
                input_pops[pop].vars["z_tilda"].view[:] = 0.0
                model.push_var_to_device(pop, 'z_tilda')

