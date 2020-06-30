import numpy as np
# from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import lif_model, LIF_PARAMS, lif_init
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init
import os

PRESENT_TIMESTEPS = 500.0
# TRIALS = 1200
TRIALS = 5

######### Set up spike source array type neuron for input population ############

# Generate poisson spike trains
poisson_spikes = []
interval = 500
freq = 5
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
target_spike_train = np.tile(target_spike_train, TRIALS)

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

# inp = model.add_neuron_population("inp", 100, "SpikeSourceArray", {},
#                                   {"startSpike": start_spike, "endSpike": end_spike})
inp = model.add_neuron_population("inp", 100, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
spikeTimes = np.hstack(poisson_spikes).astype(float)
inp.set_extra_global_param("spikeTimes", spikeTimes)
# spikeTimes needs to be set to one big vector that corresponds to all spike times of all neurons concatenated together

out = model.add_neuron_population("out", 1, lif_model, LIF_PARAMS, lif_init)
out.set_extra_global_param("spike_times", target_spike_train)

# inp2out = model.add_synapse_population("inp2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
#                                        inp, out,
#                                        superspike_model, SUPERSPIKE_PARAMS, superspike_init, {}, {},
#                                        "ExpCurr", {"tau": 5.0}, {})

inp2out = model.add_synapse_population("inp2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       inp, out,
                                       "StaticPulse", {}, {"g": 1.0}, {}, {},
                                       "DeltaCurr", {}, {})

model.build()
model.load()

######### Simulate #############

# IMG_DIR = "/home/p286814/pygenn/gennzoo/imgs"
IMG_DIR = "/home/manvi/Documents/gennzoo/imgs"

spikeTimes_view = inp.extra_global_params['spikeTimes'].view
start_spike_view = inp.vars['startSpike'].view
# err_tilda_view = out.vars["err_tilda"].view
# out_V_view = out.vars["V"].view

while model.timestep < (PRESENT_TIMESTEPS * TRIALS):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    trial = int(model.timestep // PRESENT_TIMESTEPS)

    if timestep_in_example == 0:

        if trial % 1 == 0:

            print("Trial: " + str(trial))

        if trial % 1 == 0:

            spike_ids = np.empty(0)
            spike_times = np.empty(0)

            error = np.empty(0)

            out_V = np.empty(0)

            wts = np.array([np.empty(0) for _ in range(N_INPUT)])

            # wts_sum = np.empty(0)

        if trial != 0:
            # print(type(spikeTimes))
            # print(len(spikeTimes))
            # print(type(spikeTimes[0]))
            # print(type(PRESENT_TIMESTEPS))

            spikeTimes += PRESENT_TIMESTEPS

            spikeTimes_view[:] = spikeTimes
            model.push_extra_global_param_to_device("inp", "spikeTimes")

            start_spike_view[:] = start_spike
            model.push_var_to_device("inp", "startSpike")

        # print(spikeTimes)

    model.step_time()

    if trial % 1 == 0:

        model.pull_current_spikes_from_device("inp")
        times = np.ones_like(inp.current_spikes) * model.t
        spike_ids = np.hstack((spike_ids, inp.current_spikes))
        spike_times = np.hstack((spike_times, times))

        model.pull_var_from_device("out", "err_tilda")
        error = np.hstack((error, out.vars["err_tilda"].view))

        model.pull_var_from_device("out", "V")
        out_V = np.hstack((out_V, out.vars["V"].view))

    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        model.pull_var_from_device("inp2out", "w")
        weights = inp2out.get_var_values("w")
        weights = np.reshape(weights, (weights.shape[0], 1))
        wts = np.concatenate((wts, weights), axis=1)

        # print(spike_times)
        # print(spike_ids)
        #
        # fig, axis = plt.subplots()
        # axis.scatter(spike_times, spike_ids, color="red")
        # plt.savefig("trial" + str(trial) + ".png")

        if trial % 1 == 0:

            print(error)

            # error = np.nan_to_num(error)

            timesteps = np.arange(int(PRESENT_TIMESTEPS))
            timesteps += int(PRESENT_TIMESTEPS * trial)

            # print(type(error))
            # print("error")
            # print(error)
            # print(len(error))
            # print("target spike times")
            # print(target_spike_times)
            # print(len(target_spike_times))
            # print("out_V")
            # print(out_V)
            # print(len(out_V))
            # print("spike times")
            # print(spike_times)
            # print(len(spike_times))
            # print("spike ids")
            # print(spike_ids)
            # print(len(spike_ids))

            print("Creating raster plot")

            # fig, axes = plt.subplots(4, sharex=True)
            fig, axes = plt.subplots(4, sharex=True)
            fig.tight_layout(pad=2.0)


            # print("Timesteps")
            # print(timesteps)
            # print(len(timesteps))
            axes[0].scatter(target_spike_times, [1]*len(target_spike_times))
            axes[0].set_title("Target spike train")
            axes[1].plot(timesteps, error)
            axes[1].set_title("Error")
            axes[2].plot(timesteps, out_V)
            axes[2].axhline(y=LIF_PARAMS["Vthresh"], linestyle="--", color="red")
            axes[2].set_title("Membrane potential of output neuron")
            axes[3].scatter(spike_times, spike_ids, s=10)
            axes[3].set_title("Input spikes")
            # axes[4].plot(timesteps, wts_sum)
            # axes[4].set_title("Sum of weights of synapses")

            axes[-1].set_xlabel("Time [ms]")

            save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")

            plt.savefig(save_filename)

            plt.close()

        target_spike_times += int(PRESENT_TIMESTEPS)

print("Creating weight plot")
fig, ax = plt.subplots()
wts += 0.1
wts *= (255 * 5)
print(np.amax(wts))
print(np.amin(wts))
ax.imshow(wts, cmap='gray', vmin=0, vmax=255)
ax.set_ylabel("Weights")
ax.set_xlabel("Trials")
plt.savefig("wts.png")
plt.close()
