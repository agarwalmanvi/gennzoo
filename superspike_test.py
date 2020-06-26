import numpy as np
# from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init
import os

PRESENT_TIMESTEPS = 500.0
# TRIALS = 1200
TRIALS = 2

######### Set up spike source array type neuron for input population ############

# Generate poisson spike trains
poisson_spikes = []
interval = int(PRESENT_TIMESTEPS)
freq = 8
spike_dt = 0.001
N_INPUT = 100
compare_num = freq * spike_dt
for p in range(N_INPUT):
    spike_train = np.random.random_sample(interval)
    spike_train = (spike_train < compare_num).astype(int)
    poisson_spikes.append(np.nonzero(spike_train)[0])

spike_counts = [len(n) for n in poisson_spikes]
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

########### Produce target spike train ##########

# target_spike_times = np.linspace(0, int(PRESENT_TIMESTEPS), num=4)[1:3].astype(int)
# target_spike_train = np.zeros(int(PRESENT_TIMESTEPS))
# target_spike_train[target_spike_times] = 1
# target_spike_train = np.tile(target_spike_train, TRIALS)

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

lif_model = genn_model.create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar")],
    sim_code="""
    // membrane potential dynamics
    if ($(RefracTime) <= 0.0 && $(V) >= $(Vthresh)) {
        $(V) = $(Vrest);
        $(RefracTime) = $(TauRefrac);
    }
    if ($(RefracTime) <= 0.0) {
        scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(RefracTime) -= DT;
    }
    // filtered partial derivative
    const scalar one_plus_hi = 1.0 + fabs($(beta) * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    """,
    reset_code="""
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
        ("Rmembrane", genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])())
    ],
)

LIF_PARAMS = {"C": 1.0,
              "Tau_mem": 10.0,
              "Vrest": -60.0,
              "Vthresh": -50.0,
              "Ioffset": 0.0,
              "TauRefrac": 5.0,
              "t_rise": 5.0,
              "t_decay": 10.0,
              "beta": 1.0}

lif_init = {"V": -60,
            "RefracTime": 0.0,
            "sigma_prime": 0.0}

########### Build model ################
model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = 1.0

# inp = model.add_neuron_population("inp", 100, "SpikeSourceArray", {},
#                                   {"startSpike": start_spike, "endSpike": end_spike})
inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
spikeTimes = np.hstack(poisson_spikes).astype(float)
inp.set_extra_global_param("spikeTimes", spikeTimes)
# spikeTimes needs to be set to one big vector that corresponds to all spike times of all neurons concatenated together

out = model.add_neuron_population("out", 1, lif_model, LIF_PARAMS, lif_init)
# out.set_extra_global_param("spike_times", target_spike_train)
#
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
out_voltage = out.vars['V'].view

while model.timestep < (PRESENT_TIMESTEPS * TRIALS):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % PRESENT_TIMESTEPS
    trial = int(model.timestep // PRESENT_TIMESTEPS)

    if timestep_in_example == 0:

        print("Trial: " + str(trial))

        spike_ids = np.empty(0)
        spike_times = np.empty(0)

        # z_tilda = np.array([np.empty(0) for _ in range(N_INPUT)])
        # z_tilda = np.reshape(z_tilda, (z_tilda.shape[1], z_tilda.shape[0]))

        # error = np.empty(0)
        #
        out_V = np.empty(0)
        sigma_prime = np.empty(0)
        output_spikes = np.empty(0)
        #
        # wts_sum = np.empty(0)

        out_voltage[:] = LIF_PARAMS["Vrest"]
        model.push_var_to_device('out', "V")

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

    # model.pull_var_from_device("inp", "z_tilda")
    # new_z_tilda = inp.vars["z_tilda"].view
    # new_z_tilda = np.reshape(new_z_tilda, (1, new_z_tilda.shape[0]))
    # z_tilda = np.concatenate((z_tilda, new_z_tilda))

    # model.pull_var_from_device("out", "err_tilda")
    # error = np.hstack((error, out.vars["err_tilda"].view))

    model.pull_var_from_device("out", "V")
    out_V = np.hstack((out_V, out.vars["V"].view))

    model.pull_var_from_device("out", "sigma_prime")
    sigma_prime = np.hstack((sigma_prime, out.vars["sigma_prime"].view))

    model.pull_current_spikes_from_device("out")
    out_times = np.ones_like(out.current_spikes) * model.t
    output_spikes = np.hstack((output_spikes, out_times))

    # model.pull_var_from_device("inp2out", "w")
    # weights = inp2out.get_var_values("w")
    # wts_sum = np.append(wts_sum, np.sum(weights))

    if timestep_in_example == (PRESENT_TIMESTEPS - 1):

        # error = np.nan_to_num(error)

        timesteps = np.arange(int(PRESENT_TIMESTEPS))
        timesteps += int(PRESENT_TIMESTEPS * trial)
        print("Creating raster plot")

        # fig, axes = plt.subplots(4, sharex=True)
        fig, axes = plt.subplots(3, sharex=True)
        fig.tight_layout(pad=2.0)

        # axes[0].scatter(target_spike_times, [1]*len(target_spike_times))
        # axes[0].set_title("Target spike train")
        axes[0].scatter(spike_times, spike_ids, s=10)
        axes[0].set_title("Input spikes")

        # z_tilda_0 = z_tilda[:, 0]
        # z_tilda_1 = z_tilda[:, 1]
        #
        # axes[1].plot(timesteps, z_tilda_0)
        # axes[1].set_title("z_tilda for input neuron 0")
        # axes[2].plot(timesteps, z_tilda_1)
        # axes[2].set_title("z_tilda for input neuron 1")

        axes[1].plot(timesteps, sigma_prime)
        axes[1].set_title("Sigma prime for output neuron")

        # axes[1].plot(timesteps, error)
        # axes[1].set_title("Error")
        axes[2].plot(timesteps, out_V)
        axes[2].axhline(y=LIF_PARAMS["Vthresh"], linestyle="--", color="red")
        axes[2].set_title("Membrane potential of output neuron")
        for spike_t in output_spikes:
            axes[2].axvline(x=spike_t, linestyle="--", color="red")
        # axes[3].scatter(spike_times, spike_ids, s=10)
        # axes[3].set_title("Input spikes")
        # axes[4].plot(timesteps, wts_sum)
        # axes[4].set_title("Sum of weights of synapses")

        axes[-1].set_xlabel("Time [ms]")

        save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")

        plt.savefig(save_filename)

        plt.close()

        # target_spike_times += int(PRESENT_TIMESTEPS)



