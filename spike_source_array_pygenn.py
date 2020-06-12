import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model

PRESENT_TIMESTEPS = 500.0
REPEATS = 10

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

model = genn_model.GeNNModel("float", "ssa")
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

# Build model
model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = dt

ssa = model.add_neuron_population("SSA", 100, "SpikeSourceArray", {},
                                  {"startSpike": start_spike, "endSpike": end_spike})
spikeTimes = np.hstack(poisson_spikes)
ssa.set_extra_global_param("spikeTimes", spikeTimes)

# spikeTimes needs to be set to one big vector that corresponds to all spike times of all neurons concatenated together

model.build()
model.load()

# Simulate

spikeTimes_view = ssa.extra_global_params['spikeTimes'].view
start_spike_view = ssa.vars['startSpike'].view

while model.timestep < (PRESENT_TIMESTEPS * REPEATS):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    repeat = int(model.timestep // PRESENT_TIMESTEPS)

    if timestep_in_example == 0:
        spike_ids = np.empty(0)
        spike_times = np.empty(0)

        print("Repeat: " + str(repeat))

        if repeat != 0:
            spikeTimes += PRESENT_TIMESTEPS

            spikeTimes_view[:] = spikeTimes
            model.push_extra_global_param_to_device("SSA", "spikeTimes")

            start_spike_view[:] = start_spike
            model.push_var_to_device("SSA", "startSpike")

        # print(spikeTimes)

    model.step_time()

    model.pull_current_spikes_from_device("SSA")

    times = np.ones_like(ssa.current_spikes) * model.t
    spike_ids = np.hstack((spike_ids, ssa.current_spikes))
    spike_times = np.hstack((spike_times, times))

    if timestep_in_example == (PRESENT_TIMESTEPS - 1):
        print(spike_times)
        print(spike_ids)

        fig, axis = plt.subplots()
        axis.scatter(spike_times, spike_ids, color="red")
        plt.savefig("repeat" + str(repeat) + ".png")