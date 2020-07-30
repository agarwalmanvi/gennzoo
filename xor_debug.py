import numpy as np
# from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init_classification,
                                           hidden_model, HIDDEN_PARAMS, hidden_init)
from models.synapses.superspike import superspike_model, SUPERSPIKE_PARAMS, superspike_init
import os
import random
import pickle as pkl


def create_poisson_spikes(interval, freq, spike_dt):
    """
    Create poisson spike train for 1 neuron
    :param interval: time period to create spikes for
    :param freq: spiking frequency (Hz)
    :param spike_dt: length of smallest timestep (if 1 ms, use 0.001)
    :return: spike times (np.array) within given time interval for 1 neuron
    """
    compare_num = freq * spike_dt
    spike_train = np.random.random_sample(interval)
    spike_train = (spike_train < compare_num).astype(int)
    spike_times_gen = np.nonzero(spike_train)[0]
    return spike_times_gen


STIM_FREQ = 100
wmax = SUPERSPIKE_PARAMS['wmax']

IMG_DIR = "data_xor"
STIMULUS_TIMESTEPS = 10
WAIT_TIMESTEPS = 15
ITI_RANGE = np.arange(50, 60)
TRIALS = 20001
NUM_HIDDEN = 100
WAIT_FREQ = 4

INPUT_NUM = [['time_ref', 34], ['inp0', 33], ['inp1', 33]]
N_INPUT = sum([i[1] for i in INPUT_NUM])
spike_dt = 0.001

#record= [ 0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000 ]
record= [19900,19901,19902,19903,19904,19905,19906,19907,19908,19909,19910,19911,19912,19913,19914,19915]
####### Create poisson spike trains for all input neurons and all trials ###########

# static_spikes_arr (list) stores the static poisson trains (np.array) for every neuron
# these static spike trains are used for the input presentation based on the selected stimulus
# the length of static_spikes_arr is the total number of input neurons
compare_num = STIM_FREQ * spike_dt
static_spikes = np.random.random_sample(size=(N_INPUT, STIMULUS_TIMESTEPS))
static_spikes = (static_spikes < compare_num).astype(int)
static_spikes = np.transpose(np.nonzero(static_spikes))

static_spikes_arr = []
for i in range(N_INPUT):
    if i in static_spikes[:, 0]:
        neuron_idx = np.where(static_spikes[:, 0] == i)
        neuron_spike_times = static_spikes[neuron_idx, 1]
        # print(neuron_spike_times)
        neuron_spike_times = np.reshape(neuron_spike_times, len(neuron_spike_times[0]))
        static_spikes_arr.append(neuron_spike_times)
    else:
        static_spikes_arr.append(np.array([]))

# Every trial consists of three stages: stimulus presentation, waiting time, and intertrial interval (iti)
# Here we set up a few more things for the experiment: chosen sample and iti for every trial

itis = np.random.choice(ITI_RANGE, size=TRIALS)

SAMPLES = [(0, 0, 0),
           (0, 1, 1),
           (1, 0, 1),
           (1, 1, 0)]

drawn_samples = np.random.choice(np.arange(4), size=TRIALS)

##### Build poisson spikes for all neurons ######

# poisson_spikes is a list of N_INPUT lists: each list is the spike times for each neuron
poisson_spikes = []

for neuron_idx in range(N_INPUT):
    time_elapsed = 0
    neuron_poisson_spikes = np.empty(0)

    for trial_num in range(TRIALS):

        if neuron_idx < INPUT_NUM[0][1]:

            spike_times = np.array(static_spikes_arr[neuron_idx])
            spike_times += time_elapsed
            neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        elif INPUT_NUM[0][1] <= neuron_idx < INPUT_NUM[0][1] + INPUT_NUM[1][1]:

            if SAMPLES[drawn_samples[trial_num]][0] == 1:
                spike_times = np.array(static_spikes_arr[neuron_idx])
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        else:

            if SAMPLES[drawn_samples[trial_num]][1] == 1:
                spike_times = np.array(static_spikes_arr[neuron_idx])
                spike_times += time_elapsed
                neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

        time_elapsed += STIMULUS_TIMESTEPS
        wait_plus_iti = WAIT_TIMESTEPS + itis[trial_num]

        spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, spike_dt)
        spike_times += time_elapsed
        neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
        time_elapsed += wait_plus_iti

    poisson_spikes.append(neuron_poisson_spikes)

spike_counts = [len(n) for n in poisson_spikes]
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

spikeTimes = np.hstack(poisson_spikes).astype(float)


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
    threshold_condition_code="$(startSpike) != $(endSpike) && $(t)>= $(spikeTimes)[$(startSpike)]",
    extra_global_params=[("spikeTimes", "scalar*")],
    is_auto_refractory_required=False
)

SSA_INPUT_PARAMS = {"t_rise": 5, "t_decay": 10}

ssa_input_init = {"startSpike": start_spike,
                  "endSpike": end_spike,
                  "z": 0.0,
                  "z_tilda": 0.0}

model_name= "xor_debg"

########### Build model ################
model = genn_model.GeNNModel("float", model_name)
model.dT = 1.0

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
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

print("Built main model")

######### Training #############

# Access variables during training time
out_voltage = out.vars['V'].view
out_window_of_opp = out.vars["window_of_opp"].view
out_S_pred = out.vars['S_pred'].view
out_S_miss = out.vars['S_miss'].view
inp2hid_trial_length = inp2hid.vars["trial_length"].view
hid2out_trial_length = hid2out.vars["trial_length"].view
inp2hid_trial_end_t = inp2hid.vars["trial_end_t"].view
hid2out_trial_end_t = hid2out.vars["trial_end_t"].view
hid_err_tilda = hid.vars['err_tilda'].view
out_err_tilda = out.vars['err_tilda'].view

inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view
hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view

# Data structures for recording weights at the end of every trial
wts_inp2hid = np.array([np.empty(0) for _ in range(N_INPUT * NUM_HIDDEN)])
wts_hid2out = np.array([np.empty(0) for _ in range(NUM_HIDDEN * 2)])

time_elapsed = 0

# Incorporate this into the model -- should go in the weight update model
# Random feedback
feedback_wts = np.random.normal(0.0, 1.0, size=(NUM_HIDDEN, 2))

# Record best network config encountered so far
best_wts = {'inp2hid': 0,
            'hid2out': 0}
best_err = np.inf
best_acc = 0
best_trial = 0

for trial in range(TRIALS):

    if trial % 100 == 0:
        print("Trial: " + str(trial))

    # Important to record for this trial
    target = SAMPLES[drawn_samples[trial]][-1]
    t_start = time_elapsed
    iti_chosen = itis[trial]
    total_time = STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen

    # Symmetric feedback
    #model.pull_var_from_device("hid2out", "w")
    #h2o_weights = hid2out.get_var_values("w")
    #feedback_wts = np.reshape(h2o_weights, newshape=(NUM_HIDDEN, 2))

    # Reinitialization or providing correct values for diff vars at the start of the next trial
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
    hid2out.vars['lambda'].view[:]= 0.0
    model.push_var_to_device("hid2out", "lambda")
    inp2hid.vars['lambda'].view[:]= 0.0
    model.push_var_to_device("inp2hid", "lambda")

    out_err_tilda[:] = 0.0
    model.push_var_to_device('out', 'err_tilda')

    hid_err_tilda[:] = 0.0
    model.push_var_to_device('hid', 'err_tilda')

    inp2hid_trial_length[:] = float(STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen)
    model.push_var_to_device("inp2hid", "trial_length")

    hid2out_trial_length[:] = float(STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen)
    model.push_var_to_device("hid2out", "trial_length")

    inp2hid_trial_end_t[:] = float(time_elapsed + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen - 1)
    model.push_var_to_device("inp2hid", "trial_end_t")

    hid2out_trial_end_t[:] = float(time_elapsed + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen - 1)
    model.push_var_to_device("hid2out", "trial_end_t")

    out.vars["err_rise"].view[:] = 0.0
    model.push_var_to_device('out', 'err_rise')
    out.vars["err_decay"].view[:] = 0.0
    model.push_var_to_device('out', 'err_decay')

    # Indicate the correct values for window_of_opp, S_pred, and S_miss before the stimulus is presented
    out_window_of_opp[:] = 1.0
    model.push_var_to_device("out", "window_of_opp")

    S_pred = np.array([1.0, 0.0]) if target == 0 else np.array([0.0, 1.0])
    out.vars['S_pred'].view[:] = S_pred
    model.push_var_to_device("out", "S_pred")

    out_S_miss[:] = 0.0
    model.push_var_to_device("out", "S_miss")

    # Initialize some data structures for recording various things during the trial
    if trial in record:
        out0_V = np.empty(0)
        out1_V = np.empty(0)
        out0_err = np.empty(0)
        out1_err = np.empty(0)
    
        inp_spike_ids = np.empty(0)
        inp_spike_times = np.empty(0)
    
        hid_spike_ids = np.empty(0)
        hid_spike_times = np.empty(0)

        hidztilda= np.empty((NUM_HIDDEN,0))
        
    produced_spikes = []
    err_sum = 0

    for t in range(total_time):

        model.pull_var_from_device("out", "err_tilda")
        err_output = out.vars["err_tilda"].view[:]
        err_hidden = np.sum(np.multiply(feedback_wts, err_output), axis=1)
        hid_err_tilda[:] = err_hidden
        model.push_var_to_device('hid', 'err_tilda')

        if t == STIMULUS_TIMESTEPS + WAIT_TIMESTEPS:
            out_window_of_opp[:] = 0.0
            model.push_var_to_device("out", "window_of_opp")

            if len(produced_spikes) == 0:
                out_S_miss[:] = 1.0
                model.push_var_to_device("out", "S_miss")

        model.step_time()

        if t == STIMULUS_TIMESTEPS + WAIT_TIMESTEPS:
            out_S_miss[:] = 0.0
            model.push_var_to_device("out", "S_miss")

        model.pull_current_spikes_from_device("out")
        if target in out.current_spikes:
            produced_spikes.append(model.t)

        if trial in record:
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

            model.pull_var_from_device("hid", "z_tilda")
            cur= hid.vars["z_tilda"].view[:]
            hidztilda= np.hstack((hidztilda, cur.reshape(NUM_HIDDEN,1)))
            
        err_sum += np.sum(np.abs(out.vars["err_tilda"].view[:]))

    time_elapsed += total_time

    # Record the weights at the end of the trial
    if trial in record:
        model.pull_var_from_device("inp2hid", "w")
        weights = inp2hid.get_var_values("w")
        weights= weights.reshape(N_INPUT, NUM_HIDDEN)
        #weights = np.reshape(weights, (weights.shape[0], 1))
        #wts_inp2hid = np.concatenate((wts_inp2hid, weights), axis=1)
    
        model.pull_var_from_device("hid2out", "w")
        h2o_weights = hid2out.get_var_values("w")
        h2o_weights = h2o_weights.reshape(NUM_HIDDEN, 2) 
        #weights = np.reshape(h2o_weights, (h2o_weights.shape[0], 1))
        #wts_hid2out = np.concatenate((wts_hid2out, weights), axis=1)

        ########## Make plots similar to Fig. 5b from the paper #############

        timesteps_plot = list(range(t_start, time_elapsed))
    
        num_plots = 5
    
        fig, axes = plt.subplots(num_plots, sharex=True, figsize=(15, 8))
    
        axes[0].plot(timesteps_plot, out0_err, color="royalblue")
        axes[0].plot(timesteps_plot, out1_err, color="magenta")
        axes[0].set_ylim(-1, 1)
        axes[0].set_title("Error of output neurons")
    
        axes[1].plot(timesteps_plot, out0_V, color="royalblue")
        axes[1].plot(timesteps_plot, out1_V, color="magenta")
        axes[1].set_title("Membrane voltage of output neurons")
        axes[1].axhline(y=OUTPUT_PARAMS["Vthresh"])
        axes[1].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS,
                        color="green", linestyle="--")
    
        axes[2].scatter(inp_spike_times, inp_spike_ids, s=2)
        axes[2].set_ylim(-1, N_INPUT + 1)
        axes[2].set_title("Input layer spikes")
        axes[2].axhline(y=INPUT_NUM[0][1] - 0.5, color="gray", linestyle="--")
        axes[2].axhline(y=INPUT_NUM[0][1] + INPUT_NUM[1][1] - 0.5, color="gray", linestyle="--")
        # axes[2].axvline(x=t_start + STIMULUS_TIMESTEPS, color="green", linestyle="--")
        axes[2].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS, color="green", linestyle="--")
    
        axes[3].scatter(hid_spike_times, hid_spike_ids, s=2)
        axes[3].set_ylim(-1, NUM_HIDDEN + 1)
        axes[3].set_title("Hidden layer spikes")

        for i in range(NUM_HIDDEN):
            axes[4].plot(timesteps_plot, hidztilda[i][:])
        axes[4].set_title("z_tilda hidden units")

        c = 'royalblue' if target == 0 else 'magenta'
    
        for i in range(num_plots):
            axes[i].axvspan(t_start, t_start + STIMULUS_TIMESTEPS, facecolor=c, alpha=0.3)
    
        axes[-1].set_xlabel("Time [ms]")
        # axes.set_xlabel("Time [ms]")
        x_ticks_plot = list(range(t_start, time_elapsed, 5))
        axes[-1].set_xticks(x_ticks_plot)
    
        save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
        plt.savefig(save_filename)
        plt.close()

        # Plot weights as a pixel plot with a colorbar
        print("Creating wts_inp2hid")
        plt.figure()
        plt.imshow(weights, cmap='gray', interpolation="None")
        plt.colorbar(fraction=0.05,aspect= 18)
        plt.rc('font', size= 20)
        plt.gca().invert_yaxis()
        save_filename = os.path.join(IMG_DIR, "wts_inp2hid-trial" + str(trial) + ".png")
        plt.savefig(save_filename)
        plt.close()
        print("Creating wts_hid2out")
        plt.figure()
        plt.imshow(h2o_weights, cmap='gray', interpolation="None")
        plt.colorbar(fraction=0.05,aspect= 18)
        plt.rc('font', size= 20)
        plt.gca().invert_yaxis()
        save_filename = os.path.join(IMG_DIR, "wts_hid2out-trial" + str(trial) + ".png")
        plt.savefig(save_filename)
        plt.close()

pkl_dict = {'wmax': SUPERSPIKE_PARAMS['wmax'],
            'wmin': SUPERSPIKE_PARAMS['wmin'],
            'trials': TRIALS,
            'hidden_num': NUM_HIDDEN,
            'learning_rate': SUPERSPIKE_PARAMS['r0'],
            'best_trial': best_trial,
            'feedback': 'symmetric',
            'best_acc': best_acc,
            'inp2hid': best_wts['inp2hid'],
            'hid2out': best_wts['hid2out']}

filename = os.path.join(IMG_DIR, 'config.pkl')

with open(os.path.join(IMG_DIR, "config.pkl"), 'wb') as fi:
    pkl.dump(pkl_dict, fi)

print("Complete.")
