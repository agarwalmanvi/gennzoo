import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model, OUTPUT_PARAMS, output_init,
                                           hidden_model, HIDDEN_PARAMS, hidden_init,
                                           feedback_postsyn_model)
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
                                        feedback_wts_model, feedback_wts_init)
import os
from utils import create_poisson_spikes, get_mean_square_error


"""
In this script, we will reproduce the experiment corresponding to Figure 6 from the
Superspike paper. The task is to make the output neurons produce a desired spike train
in response to a frozen noisy Poisson input spike train.
Below, we use the target spike train given in the paper's GitHub repository:
https://github.com/fzenke/pub2018superspike
The spike train is in a .ras file. The file is structured as pairs of
spike times (in seconds) and neuron indices. We first import the file and process it into
two arrays that record the spike times (in `times`) and the neuron indices (in `neurons`)
respectively. We also make a plot to have a look at what the target pattern looks like.
"""

f_path = "/home/manvi/Documents/pub2018superspike/themes/oxford-target.ras"
times = []
neurons = []
c = 0
with open(f_path) as f:
    for line in f:
        c += 1
        entry_str = line.strip()
        entry_split = entry_str.split(' ')
        times.append(float(entry_split[0]) * 1000)
        neurons.append(int(entry_split[1]))

print("Processed " + str(c) + " rows.")

# Plot the target pattern
fig, ax = plt.subplots()
ax.scatter(times, neurons, s=0.5)
plt.savefig("target_pattern.png")
plt.close()

"""
We set up some parameters to define how the simulation should work
"""
update_time = 10 * 1000 # ms
TIMESTEPS = 1890  # ms
N_INPUT = 200
N_OUTPUT = N_INPUT
TRIALS = 1501
INPUT_FREQ = 5  # Hz
TIME_FACTOR = 0.1
N_HIDDEN = 256
spike_dt = 0.001  # 1 ms
SUPERSPIKE_PARAMS['r0'] = 0.005
SUPERSPIKE_PARAMS['update_t'] = update_time

model_name_build = "pattern"
IMG_DIR = "imgs"

MODEL_BUILD_DIR = "./"

"""
We create the data structures we need in order to specify
the target spike train, repeated over several trials.
"""

target_times = np.array(times)
target_neurons = np.array(neurons)

target_poisson_spikes = []
for neuron_idx in range(N_INPUT):
    mask = target_neurons == neuron_idx
    spike_times = target_times[mask == True]
    neuron_poisson_spikes = np.empty(0)
    for trial_idx in range(TRIALS):
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        spike_times += TIMESTEPS
    target_poisson_spikes.append(neuron_poisson_spikes)

target_spike_counts = [len(n) for n in target_poisson_spikes]
target_end_spike = np.cumsum(target_spike_counts)
target_start_spike = np.empty_like(target_end_spike)
target_start_spike[0] = 0
target_start_spike[1:] = target_end_spike[0:-1]

target_spikeTimes = np.hstack(target_poisson_spikes).astype(float)

"""
We also create the spike trains we need for the
repeating Poisson input noise.
"""

poisson_spikes = []

for neuron_idx in range(N_INPUT):
    spike_times = create_poisson_spikes(TIMESTEPS, INPUT_FREQ, spike_dt, TIME_FACTOR)
    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)
    for trial_idx in range(TRIALS):
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        spike_times += TIMESTEPS
    poisson_spikes.append(neuron_poisson_spikes)

spike_counts = [len(n) for n in poisson_spikes]
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

spikeTimes = np.hstack(poisson_spikes).astype(float)

"""
Finally, we are ready to specify a custom input neuron model and the model.
"""

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

model = genn_model.GeNNModel("float", model_name_build)
model.dT = 1.0 * TIME_FACTOR

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
inp.set_extra_global_param("spikeTimes", spikeTimes)

hid = model.add_neuron_population("hid", N_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

output_init['startSpike'] = target_start_spike
output_init['endSpike'] = target_end_spike
out = model.add_neuron_population("out", N_OUTPUT, output_model, OUTPUT_PARAMS, output_init)
out.set_extra_global_param("spike_times", target_spikeTimes)

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

model.build(path_to_model=MODEL_BUILD_DIR)
model.load(path_to_model=MODEL_BUILD_DIR)

"""
Before we start the simulation, we need to specify some variables 
to make it easier for us to access model variables at simulation time.
"""

out_voltage = out.vars['V'].view
inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view
hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view
inp2hid_lambda = inp2hid.vars['lambda'].view
inp2hid_e = inp2hid.vars['e'].view
hid2out_lambda = hid2out.vars['lambda'].view
hid2out_e = hid2out.vars['e'].view
hid_err_tilda = hid.vars['err_tilda'].view
out_err_tilda = out.vars['err_tilda'].view
out2hid_wts = out2hid.vars['g'].view

wts_inp2hid = np.array([np.empty(0) for _ in range(N_INPUT * N_HIDDEN)])
wts_hid2out = np.array([np.empty(0) for _ in range(N_HIDDEN * N_OUTPUT)])

plot_interval = 50

"""
We also create all the ingredients needed to calculate the error
"""
a = 10.0
b = 5.0
tau_avg_err = 10.0
scale_tr_err_flt = 1.0 / ((((a * b) / (a - b)) ** 2) * (a / 2 + b / 2 - 2 * (a * b) / (a + b))) / tau_avg_err
mul_avgsqrerr = np.exp(-TIME_FACTOR / tau_avg_err)

avgsqrerr = np.zeros(shape=N_OUTPUT)
record_avgsqerr = np.empty(0)

"""
Finally, we will use symmetric feedback in this experiment.
Before the start of the simulation, we need to initialize the feedback
weights to be equal to the feedforward weights.
"""
model.pull_var_from_device("hid2out", "w")
h2o_weights = hid2out.get_var_values("w")
out2hid.vars['g'].view[:] = h2o_weights
model.push_var_to_device("out2hid", "g")

for trial_idx in range(TRIALS):

    print("Trial: " + str(trial_idx))

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
    hid2out_lambda[:] = 0.0
    model.push_var_to_device("hid2out", "lambda")
    inp2hid_lambda[:] = 0.0
    model.push_var_to_device("inp2hid", "lambda")
    hid2out_e[:] = 0.0
    model.push_var_to_device("hid2out", "e")
    inp2hid_e[:] = 0.0
    model.push_var_to_device("inp2hid", "e")
    out_err_tilda[:] = 0.0
    model.push_var_to_device('out', 'err_tilda')
    hid_err_tilda[:] = 0.0
    model.push_var_to_device('hid', 'err_tilda')
    out.vars["err_rise"].view[:] = 0.0
    model.push_var_to_device('out', 'err_rise')
    out.vars["err_decay"].view[:] = 0.0
    model.push_var_to_device('out', 'err_decay')
    model.pull_var_from_device("hid2out", "w")
    h2o_weights = hid2out.get_var_values("w")
    out2hid_wts[:] = h2o_weights
    model.push_var_to_device("out2hid", "g")

    if trial_idx % plot_interval == 0:
        inp_spike_ids = np.empty(0)
        inp_spike_times = np.empty(0)

        hid_spike_ids = np.empty(0)
        hid_spike_times = np.empty(0)

        out_spike_ids = np.empty(0)
        out_spike_times = np.empty(0)

    steps_in_trial = int(TIMESTEPS / TIME_FACTOR)

    for t in range(steps_in_trial):
        model.step_time()

        # Calculate learning curve
        model.pull_var_from_device("out", "err_tilda")
        temp = out_err_tilda[:]
        temp = np.power(temp, 2)
        temp = np.multiply(temp, TIME_FACTOR)
        avgsqrerr = np.multiply(avgsqrerr, mul_avgsqrerr)
        avgsqrerr = np.add(avgsqrerr, temp)

        if model.t % update_time == 0 and model.t != 0:
            error = get_mean_square_error(scale_tr_err_flt, avgsqrerr, time_elapsed, tau_avg_err)
            record_avgsqerr = np.hstack((record_avgsqerr, error))
            avgsqrerr = np.zeros(shape=N_OUTPUT)

            # Record the weights after they have been updated
            model.pull_var_from_device("inp2hid", "w")
            weights = inp2hid.get_var_values("w")
            weights = np.reshape(weights, (weights.shape[0], 1))
            wts_inp2hid = np.concatenate((wts_inp2hid, weights), axis=1)

            model.pull_var_from_device("hid2out", "w")
            h2o_weights = hid2out.get_var_values("w")
            weights = np.reshape(h2o_weights, (h2o_weights.shape[0], 1))
            wts_hid2out = np.concatenate((wts_hid2out, weights), axis=1)

            """
            Since the feedforward weights have been updated, we need to
            also update the feedback weights since we are using symmetric feedback
            """
            model.pull_var_from_device("hid2out", "w")
            h2o_weights = hid2out.get_var_values("w")
            out2hid.vars['g'].view[:] = h2o_weights
            model.push_var_to_device("out2hid", "g")

        if trial_idx % plot_interval == 0:

            model.pull_current_spikes_from_device("inp")
            times = np.ones_like(inp.current_spikes) * model.t
            inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
            inp_spike_times = np.hstack((inp_spike_times, times))

            model.pull_current_spikes_from_device("hid")
            times = np.ones_like(hid.current_spikes) * model.t
            hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
            hid_spike_times = np.hstack((hid_spike_times, times))

            model.pull_current_spikes_from_device("out")
            times = np.ones_like(out.current_spikes) * model.t
            out_spike_ids = np.hstack((out_spike_ids, out.current_spikes))
            out_spike_times = np.hstack((out_spike_times, times))

    if trial_idx % plot_interval == 0:

        fig, axes = plt.subplots(3, figsize=(15, 10), sharex=True)

        point_size = 2.0

        axes[0].scatter(inp_spike_times, inp_spike_ids, s=point_size)
        axes[0].set_ylim(-1, N_INPUT + 1)
        axes[0].set_title("Input layer spikes")

        axes[1].scatter(hid_spike_times, hid_spike_ids, s=point_size)
        axes[1].set_ylim(-1, N_HIDDEN + 1)
        axes[1].set_title("Hidden layer spikes")

        axes[2].scatter(out_spike_times, out_spike_ids, s=point_size)
        axes[2].set_ylim(-1, N_OUTPUT + 1)
        axes[2].set_title("Output layer spikes")

        axes[-1].set_xlabel("Time [ms]")
        xticks = list(range((trial_idx * TIMESTEPS), ((trial_idx + 1) * TIMESTEPS), 200))
        axes[-1].set_xticks(xticks)

        save_filename = os.path.join(IMG_DIR, "trial" + str(trial_idx) + ".png")
        plt.savefig(save_filename)
        plt.close()

"""
Just as in the script for xor and the timed spikes experiments, 
we can choose to use the weights or the error by making a plot
or dumping the values in a pickle file.
"""

