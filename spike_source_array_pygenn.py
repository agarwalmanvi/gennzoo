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
dt = 1.0
rate = 10.0
isi = 1000.0 / (rate * dt)
for p in range(100):
    time = 0.0
    neuron_spikes = []
    while True:
        time += expon.rvs(1) * isi
        if time >= 500.0:
            break
        else:
            neuron_spikes.append(time)
    poisson_spikes.append(neuron_spikes)

# poisson_spikes is a list of 100 lists: each list is the spike times for each neuron

model = genn_model.GeNNModel("float", "superspike_test")
model.dT = dt

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

########### Build model ################
model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = dt

inp = model.add_neuron_population("inp", 100, "SpikeSourceArray", {},
                                  {"startSpike": start_spike, "endSpike": end_spike})
spikeTimes = np.hstack(poisson_spikes)
inp.set_extra_global_param("spikeTimes", spikeTimes)
# spikeTimes needs to be set to one big vector that corresponds to all spike times of all neurons concatenated together

out = model.add_neuron_population("out", 1, lif_model, LIF_PARAMS, lif_init)
out.set_extra_global_params("spike_times", target_spike_train)

inp2out = model.add_synapse_population("inp2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       inp, out,
                                       superspike_model, SUPERSPIKE_PARAMS, superspike_init, {}, {},
                                       "ExpCurr", {"tau": 5.0}, {})

model.build()
model.load()

######### Simulate #############

spikeTimes_view = inp.extra_global_params['spikeTimes'].view
start_spike_view = inp.vars['startSpike'].view
err_tilda_view = out.vars["err_tilda"].view
out_V_view = out.vars["V"].view

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
    error = np.hstack((error, err_tilda_view))

    model.pull_var_from_device("out", "V")
    out_V = np.hstack((out_V, out_V_view))

    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # print(spike_times)
        # print(spike_ids)
        #
        # fig, axis = plt.subplots()
        # axis.scatter(spike_times, spike_ids, color="red")
        # plt.savefig("trial" + str(trial) + ".png")

        print("Creating raster plot")

        fig, axes = plt.subplots(3, sharex=True)

        timesteps = list(range(PRESENT_TIMESTEPS))
        axes[0].plot(timesteps, error)
        axes[1].plot(timesteps, out_V)
        axes[2].scatter(spike_times, spike_ids)