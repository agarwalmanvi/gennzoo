import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model_classification, OUTPUT_PARAMS, output_init_classification,
										   hidden_model, HIDDEN_PARAMS, hidden_init,
										   feedback_postsyn_model)
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
										feedback_wts_model, feedback_wts_init)
import os
from utils import create_poisson_spikes, get_mean_square_error
from math import ceil
import configparser
import sys

if len(sys.argv) > 1:
	cfg_file = sys.argv[1]
	config = configparser.ConfigParser()
	config.read(cfg_file)
	experiment_type = str(config["Superspike"]["ExperimentType"])
	learning_rate = float(config["Superspike"]["LearningRate"])
	TRIALS = int(config["Superspike"]["Trials"])
	SUPERSPIKE_PARAMS["r0"] = learning_rate

else:
	print("Using default arguments")
	experiment_type = "default"
	learning_rate = SUPERSPIKE_PARAMS["r0"]
	TRIALS = 1001

model_name = experiment_type + "_" + str(learning_rate)
# IMG_DIR = os.path.join("/data", "p286814", "imgs", experiment_type, learning_rate)
IMG_DIR = "imgs"
# MODEL_BUILD_DIR = os.environ.get('TMPDIR') + os.path.sep
MODEL_BUILD_DIR = "./"

TIME_FACTOR = 0.1
UPDATE_TIME = 500  # ms

STIMULUS_TIMESTEPS = 10  # ms
WAIT_TIMESTEPS = 15  # ms
ITI_RANGE = np.arange(50, 60)  # ms
TEST_ITI = 55  # ms

N_HIDDEN = 100
N_OUTPUT = 2
WAIT_FREQ = 4  # Hz
STIM_FREQ = 100  # Hz

INPUT_NUM = [['time_ref', 34], ['inp0', 33], ['inp1', 33]]
N_INPUT = sum([i[1] for i in INPUT_NUM])
SPIKE_DT = 0.001  # 1 ms

SUPERSPIKE_PARAMS["update_t"] = UPDATE_TIME

####### Create poisson spike trains for all input neurons and all trials ###########

"""
`static_spikes_arr` (list) stores the static poisson trains (np.array) for every neuron.
These static spike trains are used for the input presentation based on the selected stimulus.
The length of `static_spikes_arr` is the total number of input neurons `N_INPUT`.
"""

compare_num = STIM_FREQ * (SPIKE_DT * TIME_FACTOR)
static_spikes = np.random.random_sample(size=(N_INPUT, int(STIMULUS_TIMESTEPS / TIME_FACTOR)))
static_spikes = (static_spikes < compare_num).astype(int)
static_spikes = np.transpose(np.nonzero(static_spikes))

static_spikes_arr = []
for i in range(N_INPUT):
	if i in static_spikes[:, 0]:
		neuron_idx = np.where(static_spikes[:, 0] == i)
		neuron_spike_times = static_spikes[neuron_idx, 1]
		# print(neuron_spike_times)
		neuron_spike_times = np.reshape(neuron_spike_times, len(neuron_spike_times[0]))
		neuron_spike_times = np.multiply(neuron_spike_times, TIME_FACTOR)
		static_spikes_arr.append(neuron_spike_times)
	else:
		static_spikes_arr.append(np.array([]))

"""
Every trial consists of three stages: 
stimulus presentation, waiting time, and an intertrial interval (iti).
Here we set up a few more things for the experiment: 
for every trial, we specify the chosen sample and the iti
"""

itis = np.random.choice(ITI_RANGE, size=TRIALS)

SAMPLES = [(0, 0, 0),
		   (0, 1, 1),
		   (1, 0, 1),
		   (1, 1, 0)]

drawn_samples = np.random.choice(np.arange(4), size=TRIALS)

"""
Before we start running the simulation, we will compile all spike times for all neurons
across all trials. This means that the spike trains are not built dynamically but
have to be set before the simulation begins.
`poisson_spikes` will be a list of `N_INPUT` lists: each list is the spike times for each neuron.
"""

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

		spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, SPIKE_DT, TIME_FACTOR)
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

"""
Just as we needed to fix the training spike trains for all neurons before
starting the simulation, we also need to fix the testing spike trains
for all neurons. 
"""
test_poisson_spikes = []

for neuron_idx in range(N_INPUT):

	time_elapsed = 0
	neuron_poisson_spikes = np.empty(0)

	for sample_idx in range(len(SAMPLES)):

		if neuron_idx < INPUT_NUM[0][1]:

			spike_times = np.array(static_spikes_arr[neuron_idx])
			spike_times += time_elapsed
			neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

		elif INPUT_NUM[0][1] <= neuron_idx < INPUT_NUM[0][1] + INPUT_NUM[1][1]:

			if SAMPLES[sample_idx][0] == 1:
				spike_times = np.array(static_spikes_arr[neuron_idx])
				spike_times += time_elapsed
				neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

		else:

			if SAMPLES[sample_idx][1] == 1:
				spike_times = np.array(static_spikes_arr[neuron_idx])
				spike_times += time_elapsed
				neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))

		time_elapsed += STIMULUS_TIMESTEPS
		wait_plus_iti = WAIT_TIMESTEPS + TEST_ITI

		spike_times = create_poisson_spikes(wait_plus_iti, WAIT_FREQ, SPIKE_DT, TIME_FACTOR)
		spike_times += time_elapsed
		neuron_poisson_spikes = np.hstack((neuron_poisson_spikes, spike_times))
		time_elapsed += wait_plus_iti

	test_poisson_spikes.append(neuron_poisson_spikes)

test_spike_counts = [len(n) for n in test_poisson_spikes]
test_end_spike = np.cumsum(test_spike_counts)
test_start_spike = np.empty_like(test_end_spike)
test_start_spike[0] = 0
test_start_spike[1:] = test_end_spike[0:-1]

test_spikeTimes = np.hstack(test_poisson_spikes).astype(float)

"""
Below, we modified the in-built `SpikeSourceArray` model to include the `z` variable
"""

ssa_input_model = genn_model.create_custom_neuron_class(
	"ssa_input_model",
	param_names=["t_rise", "t_decay"],
	var_name_types=[("startSpike", "unsigned int"), ("endSpike", "unsigned int"),
					("z", "scalar"), ("z_tilda", "scalar")],
	sim_code="""
	// filtered presynaptic trace
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

"""
We also define some initializations for the input population in the testing model
"""

test_ssa_input_init = {"startSpike": test_start_spike,
					   "endSpike": test_end_spike}

test_lif_init = {"V": -60.0,
				 "RefracTime": 0.0}

TEST_LIF_PARAMS = {"C": 10.0,
				   "TauM": 10.0,
				   "Vrest": -60.0,
				   "Vreset": -60.0,
				   "Vthresh": -50.0,
				   "Ioffset": 0.0,
				   "TauRefrac": 5.0}

"""
We're ready to build our model!
We define three populations: input, hidden, and output.
We make feedforward connections from input to hidden and hidden to output.
We also make feedback connections from output to hidden.
"""
model = genn_model.GeNNModel(precision="float", model_name=model_name, time_precision="double")
model.dT = 1.0 * TIME_FACTOR

inp = model.add_neuron_population("inp", N_INPUT, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
inp.set_extra_global_param("spikeTimes", spikeTimes)

hid = model.add_neuron_population("hid", N_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

out = model.add_neuron_population("out", N_OUTPUT, output_model_classification, OUTPUT_PARAMS,
								  output_init_classification)

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
We use Xavier initialization to set the weights of the inp2hid and hid2out connections
"""
a = 1.0 / np.sqrt(N_INPUT)
inp2hid_wt_init = np.random.uniform(low=-a, high=a, size=N_INPUT*N_HIDDEN)
inp2hid.vars["w"].view[:] = inp2hid_wt_init
model.push_var_to_device("inp2hid", "w")
a = 1.0 / np.sqrt(N_HIDDEN)
hid2out_wt_init = np.random.uniform(low=-a, high=a, size=N_HIDDEN*N_OUTPUT)
hid2out.vars["w"].view[:] = hid2out_wt_init
model.push_var_to_device("hid2out", "w")


"""
Before we start the simulation, we need to define some shorthands that
we can use to access different model variables at simulation time. 
"""
out_voltage = out.vars['V'].view
out_window_of_opp = out.vars["window_of_opp"].view
out_S_pred = out.vars['S_pred'].view
out_S_miss = out.vars['S_miss'].view
hid_err_tilda = hid.vars['err_tilda'].view
out_err_tilda = out.vars['err_tilda'].view
inp_z = inp.vars['z'].view
inp_z_tilda = inp.vars["z_tilda"].view
hid_z = hid.vars['z'].view
hid_z_tilda = hid.vars['z_tilda'].view
hid_voltage = hid.vars['V'].view
inp2hid_lambda = inp2hid.vars['lambda'].view
hid2out_lambda = hid2out.vars['lambda'].view
inp2hid_e = inp2hid.vars['e'].view
hid2out_e = hid2out.vars['e'].view
out_err_rise = out.vars["err_rise"].view
out_err_decay = out.vars["err_decay"].view

"""
We will also create some data structures that can record the weights
of the inp2hid and hid2out connections. These can be used to plot how the weights change.
"""
wts_inp2hid = np.array([np.empty(0) for _ in range(N_INPUT * N_HIDDEN)])
wts_hid2out = np.array([np.empty(0) for _ in range(N_HIDDEN * N_OUTPUT)])

"""
Here we will use random feedback. The feedback weights need to be set and pushed
to the model only once at the start of the simulation, which we do below.
"""
feedback_wts = np.random.normal(0.0, 1.0, size=(N_HIDDEN, N_OUTPUT)).flatten()
out2hid.vars['g'].view[:] = feedback_wts
model.push_var_to_device('out2hid', 'g')

"""
Since we will also be testing the progress of our network, we will need some
data structures to record the best performance encountered so far as well as
the corresponding network configuration. 
"""
best_wts = {'inp2hid': 0,
			'hid2out': 0}
best_err = np.inf
best_acc = 0
best_trial = 0

"""
Finally, to plot the learning curve, we need to define the variables and data structures as given below.
"""
a = 10.0
b = 5.0
tau_avg_err = 10.0
scale_tr_err_flt = 1.0 / ((((a * b) / (a - b)) ** 2) * (a / 2 + b / 2 - 2 * (a * b) / (a + b))) / tau_avg_err
mul_avgsqrerr = np.exp(-TIME_FACTOR / tau_avg_err)
record_avgsqerr = np.empty(0)
avgsqrerr = np.zeros(shape=2)

time_elapsed = 0  # ms
plot_interval = 1

for trial in range(TRIALS):

	if trial % 1 == 0:
		print("\n")
		print("Trial: " + str(trial))

	# We need some information about this trial...
	target = SAMPLES[drawn_samples[trial]][-1]
	t_start = time_elapsed  # ms
	iti_chosen = itis[trial]  # ms
	total_time = STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + iti_chosen  # ms

	# Reinitialize or provide correct values for different variables at the start of the next trial
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
	out_err_rise[:] = 0.0
	model.push_var_to_device('out', 'err_rise')
	out_err_decay[:] = 0.0
	model.push_var_to_device('out', 'err_decay')

	# Indicate the correct values for window_of_opp, S_pred, and S_miss before the stimulus is presented
	out_window_of_opp[:] = 1.0
	model.push_var_to_device("out", "window_of_opp")

	S_pred = np.array([1.0, 0.0]) if target == 0 else np.array([0.0, 1.0])
	out.vars['S_pred'].view[:] = S_pred
	model.push_var_to_device("out", "S_pred")

	out_S_miss[:] = 0.0
	model.push_var_to_device("out", "S_miss")

	# To plot the behaviour of the network, we need some data structures to record during the trial
	if trial % plot_interval == 0:
		out0_V = np.empty(0)
		out1_V = np.empty(0)

		out0_err = np.empty(0)
		out1_err = np.empty(0)

		inp_spike_ids = np.empty(0)
		inp_spike_times = np.empty(0)

		hid_spike_ids = np.empty(0)
		hid_spike_times = np.empty(0)

	produced_spikes = []

	steps = int(total_time / TIME_FACTOR)

	for t in range(steps):

		if t == ((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
			out_window_of_opp[:] = 0.0
			model.push_var_to_device("out", "window_of_opp")

			if len(produced_spikes) == 0:
				out_S_miss[:] = 1.0
				model.push_var_to_device("out", "S_miss")

		model.step_time()

		if t == ((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
			out_S_miss[:] = 0.0
			model.push_var_to_device("out", "S_miss")

		model.pull_current_spikes_from_device("out")
		if target in out.current_spikes:
			produced_spikes.append(model.t)

		# At each step, we need to get the error for each output neuron
		model.pull_var_from_device("out", "err_tilda")
		temp = out_err_tilda[:]
		temp = np.power(temp, 2)
		temp = np.multiply(temp, TIME_FACTOR)
		avgsqrerr = np.multiply(avgsqrerr, mul_avgsqrerr)
		avgsqrerr = np.add(avgsqrerr, temp)

		# Record the previous weights in case you can need to test
		if np.round(model.t + TIME_FACTOR, decimals=1) % UPDATE_TIME == 0:
			model.pull_var_from_device("inp2hid", "w")
			prev_inp2hid = inp2hid.get_var_values("w")
			model.pull_var_from_device("hid2out", "w")
			prev_hid2out = hid2out.get_var_values("w")

		# At the time step when the model has updated the weights
		if model.t % UPDATE_TIME == 0 and model.t != 0:
			# We calculate the total average square error
			error = get_mean_square_error(scale_tr_err_flt, avgsqrerr, time_elapsed, tau_avg_err)
			record_avgsqerr = np.hstack((record_avgsqerr, error))
			avgsqrerr = np.zeros(shape=2)

			# We pull the weights and record it in our data structure
			model.pull_var_from_device("inp2hid", "w")
			weights = inp2hid.get_var_values("w")
			weights = np.reshape(weights, (weights.shape[0], 1))
			wts_inp2hid = np.concatenate((wts_inp2hid, weights), axis=1)

			model.pull_var_from_device("hid2out", "w")
			h2o_weights = hid2out.get_var_values("w")
			h2o_weights = np.reshape(h2o_weights, (h2o_weights.shape[0], 1))
			wts_hid2out = np.concatenate((wts_hid2out, h2o_weights), axis=1)

			# We also test the network configuration
			if error <= best_err:

				print("Testing in trial: " + str(trial))

				best_err = error

				test_network = genn_model.GeNNModel(precision="float", model_name=model_name+"_test",
													time_precision="double")
				test_network.dT = 1.0 * TIME_FACTOR

				test_inp = test_network.add_neuron_population("inp", N_INPUT, "SpikeSourceArray", {},
															  test_ssa_input_init)
				test_inp.set_extra_global_param("spikeTimes", test_spikeTimes)

				test_hid = test_network.add_neuron_population("hid", N_HIDDEN, "LIF", TEST_LIF_PARAMS, test_lif_init)

				test_out = test_network.add_neuron_population("out", N_OUTPUT, "LIF", TEST_LIF_PARAMS,
															  test_lif_init)

				test_inp2hid = test_network.add_synapse_population("inp2hid", "DENSE_INDIVIDUALG",
																   genn_wrapper.NO_DELAY,
																   test_inp, test_hid,
																   "StaticPulse", {}, {"g": prev_inp2hid.flatten()}, {},
																   {},
																   "ExpCurr", {"tau": 5.0}, {})

				test_hid2out = test_network.add_synapse_population("hid2out", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
																   test_hid, test_out,
																   "StaticPulse", {}, {"g": prev_hid2out.flatten()}, {}, {},
																   "ExpCurr", {"tau": 5.0}, {})

				test_network.build(path_to_model=MODEL_BUILD_DIR)
				test_network.load(path_to_model=MODEL_BUILD_DIR)

				num_correct = 0

				for sample_idx in range(len(SAMPLES)):

					test_target = SAMPLES[sample_idx][-1]
					test_non_target = 1 - test_target

					test_target_spikes = []
					test_non_target_spikes = []

					test_steps = int((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS + TEST_ITI) / TIME_FACTOR)

					for test_t in range(test_steps):

						test_network.step_time()

						if test_t < int((STIMULUS_TIMESTEPS + WAIT_TIMESTEPS) / TIME_FACTOR):
							test_network.pull_current_spikes_from_device("out")
							if test_target in test_out.current_spikes:
								test_target_spikes.append(test_network.t)
							if test_non_target in test_out.current_spikes:
								test_non_target_spikes.append(test_network.t)

					if len(test_target_spikes) != 0 and len(test_non_target_spikes) == 0:
						num_correct += 1

				accuracy = num_correct / 4

				if accuracy > best_acc:
					test_network.pull_var_from_device("inp2hid", "g")
					best_wts['inp2hid'] = test_inp2hid.get_var_values("g")
					test_network.pull_var_from_device("hid2out", "g")
					best_wts['hid2out'] = test_hid2out.get_var_values("g")
					best_acc = accuracy
					best_trial = trial

		if trial % plot_interval == 0:
			model.pull_current_spikes_from_device("inp")
			times = np.ones_like(inp.current_spikes) * model.t
			inp_spike_ids = np.hstack((inp_spike_ids, inp.current_spikes))
			inp_spike_times = np.hstack((inp_spike_times, times))

			model.pull_current_spikes_from_device("hid")
			times = np.ones_like(hid.current_spikes) * model.t
			hid_spike_ids = np.hstack((hid_spike_ids, hid.current_spikes))
			hid_spike_times = np.hstack((hid_spike_times, times))

			model.pull_var_from_device("out", "V")
			out0_V = np.hstack((out0_V, out.vars["V"].view[0]))
			out1_V = np.hstack((out1_V, out.vars["V"].view[1]))

			model.pull_var_from_device("out", "err_tilda")
			out0_err = np.hstack((out0_err, out.vars["err_tilda"].view[0]))
			out1_err = np.hstack((out1_err, out.vars["err_tilda"].view[1]))

	time_elapsed += total_time

	"""
	At the end of the trial, we can make a plot
	"""
	if trial % plot_interval == 0:
		timesteps_plot = np.linspace(t_start, time_elapsed, num=steps)

		num_plots = 4

		fig, axes = plt.subplots(num_plots, sharex=True, figsize=(15, 8))

		axes[0].plot(timesteps_plot, out0_err, color="royalblue")
		axes[0].plot(timesteps_plot, out1_err, color="magenta")
		axes[0].set_ylim(-1.1, 1.1)
		axes[0].set_title("Error of output neurons")

		axes[1].plot(timesteps_plot, out0_V, color="royalblue")
		axes[1].plot(timesteps_plot, out1_V, color="magenta")
		axes[1].set_title("Membrane voltage of output neurons")
		axes[1].axhline(y=OUTPUT_PARAMS["Vthresh"])
		axes[1].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS,
						color="green", linestyle="--")

		axes[2].scatter(inp_spike_times, inp_spike_ids)
		axes[2].set_ylim(-1, N_INPUT + 1)
		axes[2].set_title("Input layer spikes")
		axes[2].axhline(y=INPUT_NUM[0][1] - 0.5, color="gray", linestyle="--")
		axes[2].axhline(y=INPUT_NUM[0][1] + INPUT_NUM[1][1] - 0.5, color="gray", linestyle="--")
		axes[2].axvline(x=t_start + STIMULUS_TIMESTEPS + WAIT_TIMESTEPS, color="green", linestyle="--")

		axes[3].scatter(hid_spike_times, hid_spike_ids)
		axes[3].set_ylim(-1, N_HIDDEN + 1)
		axes[3].set_title("Hidden layer spikes")

	c = 'royalblue' if target == 0 else 'magenta'

	for i in range(num_plots):
		axes[i].axvspan(t_start, t_start + STIMULUS_TIMESTEPS, facecolor=c, alpha=0.3)

	axes[-1].set_xlabel("Time [ms]")
	x_ticks_plot = list(range(t_start, time_elapsed, int(ceil(5 * TIME_FACTOR))))
	axes[-1].set_xticks(x_ticks_plot)

	save_filename = os.path.join(IMG_DIR, "trial" + str(trial) + ".png")
	plt.savefig(save_filename)
	plt.close()

"""
With this script, we should see some plots in the `IMG_DIR`.
We can also add some code below to dump all the variables that we have been keeping
track of through the simulation, such as the weights or the best configuration after testing
or the history of errors.
"""