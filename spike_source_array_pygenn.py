import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import lif_model, LIF_PARAMS, lif_init
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init

PRESENT_TIMESTEPS = 500.0
TRIALS = 1

######### Set up spike source array type neuron for input population ############

# Generate poisson spike trains
poisson_spikes = []
interval = 500
freq = 1
spike_dt = 0.001
N_INPUT = 100
compare_num = freq * spike_dt
for p in range(N_INPUT):
    spike_train = np.random.random_sample(interval)
    spike_train = (spike_train < compare_num).astype(int)
    poisson_spikes.append(np.nonzero(spike_train)[0])
# dt = 1.0
# rate = 10.0
# isi = 1000.0 / (rate * dt)
# for p in range(100):
#     time = 0.0
#     neuron_spikes = []
#     while True:
#         time += expon.rvs(1) * isi
#         if time >= 500.0:
#             break
#         else:
#             neuron_spikes.append(time)
#     poisson_spikes.append(neuron_spikes)

# poisson_spikes is a list of 100 lists: each list is the spike times for each neuron

# Count spikes each neuron should emit
spike_counts = [len(n) for n in poisson_spikes]

# spike_counts is a list of 100 elements, each element corresponding to the number of spikes each neuron emits

# Get start and end indices of each spike sources section
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

# start_spike and end_spike gives the positions at which index each neuron's spike times starts and ends
# e.g. start_spike[0] is the starting index and end_spike[0] is the ending index of the first neuron's spike times

########### Produce target spike train ##########

target_spike_times = np.linspace(0, 500, num=7)[1:6].astype(int)
target_spike_train = np.zeros(int(PRESENT_TIMESTEPS))
target_spike_train[target_spike_times] = 1

########### Custom spike source array neuron model ############

ssa_input_model = genn_model.create_custom_neuron_class(
    "ssa_input_model",
    param_names=["t_rise", "t_decay"],
    var_name_types=[("startSpike", "unsigned int"), ("endSpike", "unsigned int"),
                    ("z", "scalar"), ("z_tilda", "scalar")],
    sim_code="""
    // filtered presynaptic trace
    $(z) *= exp(- DT / $(t_rise));
    $(z_tilda) += ((- $(z_tilda) + $(z)) / $(t_decay)) * DT;
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

# inp = model.add_neuron_population("inp", 100, "SpikeSourceArray", {},
#                                   {"startSpike": start_spike, "endSpike": end_spike})
inp = model.add_neuron_population("inp", 100, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
spikeTimes = np.hstack(poisson_spikes)
inp.set_extra_global_param("spikeTimes", spikeTimes)
# spikeTimes needs to be set to one big vector that corresponds to all spike times of all neurons concatenated together

out = model.add_neuron_population("out", 1, lif_model, LIF_PARAMS, lif_init)
out.set_extra_global_param("spike_times", target_spike_train)

inp2out = model.add_synapse_population("inp2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       inp, out,
                                       superspike_model, SUPERSPIKE_PARAMS, superspike_init, {}, {},
                                       "ExpCurr", {"tau": 5.0}, {})

model.build()
model.load()

######### Simulate #############

spikeTimes_view = inp.extra_global_params['spikeTimes'].view
start_spike_view = inp.vars['startSpike'].view
# err_tilda_view = out.vars["err_tilda"].view
# out_V_view = out.vars["V"].view

while model.timestep < (PRESENT_TIMESTEPS * TRIALS):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    trial = int(model.timestep // PRESENT_TIMESTEPS)

    if timestep_in_example == 0:

        print("Trial: " + str(trial))

        spike_ids = np.empty(0)
        spike_times = np.empty(0)

        error = np.empty(0)

        out_V = np.empty(0)

        wts_sum = np.empty(0)

        if trial != 0:
            spikeTimes += PRESENT_TIMESTEPS

            spikeTimes_view[:] = spikeTimes
            model.push_extra_global_param_to_device("inp", "spikeTimes")

            start_spike_view[:] = start_spike
            model.push_var_to_device("inp", "startSpike")

        # print(spikeTimes)

    model.step_time()

    model.pull_current_spikes_from_device("inp")
    times = np.ones_like(inp.current_spikes) * model.t
    spike_ids = np.hstack((spike_ids, inp.current_spikes))
    spike_times = np.hstack((spike_times, times))

    model.pull_var_from_device("out", "err_tilda")
    error = np.hstack((error, out.vars["err_tilda"].view))

    model.pull_var_from_device("out", "V")
    out_V = np.hstack((out_V, out.vars["V"].view))

    model.pull_var_from_device("inp2out", "w")
    weights = inp2out.get_var_values("w")
    wts_sum = np.append(wts_sum, np.sum(weights))

    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # print(spike_times)
        # print(spike_ids)
        #
        # fig, axis = plt.subplots()
        # axis.scatter(spike_times, spike_ids, color="red")
        # plt.savefig("trial" + str(trial) + ".png")

        print("Creating raster plot")

        fig, axes = plt.subplots(4, sharex=True)
        fig.tight_layout(pad=2.0)

        timesteps = list(range(int(PRESENT_TIMESTEPS)))
        axes[0].plot(timesteps, error)
        axes[0].scatter(target_spike_times, [0.03]*len(target_spike_times))
        axes[0].set_title("Error")
        axes[1].plot(timesteps, out_V)
        axes[1].set_title("Membrane potential of output neuron")
        axes[2].scatter(spike_times, spike_ids, s=20)
        axes[2].set_title("Input spikes")
        axes[3].plot(timesteps, wts_sum)
        axes[3].set_title("Sum of weights of synapses")

        axes[-1].set_xlabel("Time [ms]")

        plt.savefig("trial" + str(trial) + ".png")