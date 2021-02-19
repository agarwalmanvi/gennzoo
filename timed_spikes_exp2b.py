import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model
from models.neurons.lif_superspike import (output_model, OUTPUT_PARAMS, output_init,
                                           hidden_model, HIDDEN_PARAMS, hidden_init,
                                           feedback_postsyn_model)
from models.neurons.ssa_input import ssa_input_model, ssa_input_init, SSA_INPUT_PARAMS
from models.synapses.superspike import (superspike_model, SUPERSPIKE_PARAMS, superspike_init,
                                        feedback_wts_model, feedback_wts_init)
import os
from models.parameters import r0, tau_avg_err
import pickle as pkl
from utils import create_poisson_spikes, get_mean_square_error

"""
In this script, we are going to recreate the experiment from Fig. 4 (c,d,e) of the Superspike paper.
We reuse code from timed_spikes_exp2a.py
"""

TRIALS = 1600

for feedback_type in ["random", "symmetric", "uniform"]:

    print("Feedback type: " + feedback_type)

    # avgsqerr_arr is a matrix with row=simulation number and col=trial number
    avgsqerr_arr = np.zeros(shape=(20, TRIALS))

    for sim_idx in range(20):

        print("Simulation: " + str(sim_idx))

        PRESENT_TIMESTEPS = 500.0
        NUM_HIDDEN = 4
        SUPERSPIKE_PARAMS["update_t"] = PRESENT_TIMESTEPS
        TIME_FACTOR = 0.1

        """
        First we create the poisson spike trains for all the input neurons.
        Below, `poisson_spikes` is a list of 100 lists: each list is the spike times for each neuron.
        We create `start_spike` and `end_spike`, which we need to initialize the input layer.
        `start_spike` and `end_spike` give the indices at which each neuron's spike times starts and ends
        e.g. start_spike[0] is the starting index and end_spike[0] is the ending index of the 0th neuron's spike times.
        """

        poisson_spikes = []
        freq = 8
        spike_dt = 0.001
        N_INPUT = 100
        interval = int(PRESENT_TIMESTEPS)
        for p in range(N_INPUT):
            neuron_spike_train = create_poisson_spikes(interval, freq, spike_dt, 1.0)
            poisson_spikes.append(neuron_spike_train)

        spike_counts = [len(n) for n in poisson_spikes]

        end_spike = np.cumsum(spike_counts)
        start_spike = np.empty_like(end_spike)
        start_spike[0] = 0
        start_spike[1:] = end_spike[0:-1]

        spikeTimes = np.hstack(poisson_spikes).astype(float)

        """
        The target spike train is a series of 5 equidistant spikes, which we create below.
        """

        base_target_spike_times = np.linspace(0, 500, num=7)[1:6].astype(int)
        add_arr = np.arange(TRIALS)
        add_arr = np.repeat(add_arr, len(base_target_spike_times))
        add_arr = np.multiply(add_arr, interval)
        target_spike_times = np.tile(base_target_spike_times, TRIALS)
        target_spike_times = np.add(target_spike_times, add_arr)

        target_poisson_spikes = [target_spike_times]
        spike_counts = [len(n) for n in target_poisson_spikes]
        target_end_spike = np.cumsum(spike_counts)
        target_start_spike = np.empty_like(target_end_spike)
        target_start_spike[0] = 0
        target_start_spike[1:] = target_end_spike[0:-1]

        target_spikeTimes = np.hstack(target_poisson_spikes).astype(float)

        # Set up startSpike and endSpike for custom SpikeSourceArray model

        ssa_input_init["startSpike"] = start_spike
        ssa_input_init["endSpike"] = end_spike

        ########### Build model ################
        model = genn_model.GeNNModel("float", "spike_source_array", time_precision="double")
        model.dT = 1.0 * TIME_FACTOR

        inp = model.add_neuron_population("inp", 100, ssa_input_model, SSA_INPUT_PARAMS, ssa_input_init)
        inp.set_extra_global_param("spikeTimes", spikeTimes)

        hid = model.add_neuron_population("hid", NUM_HIDDEN, hidden_model, HIDDEN_PARAMS, hidden_init)

        output_init['startSpike'] = target_start_spike
        output_init['endSpike'] = target_end_spike
        out = model.add_neuron_population("out", 1, output_model, OUTPUT_PARAMS, output_init)
        out.set_extra_global_param("spikeTimes", target_spikeTimes)

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

        model.build()
        model.load()

        """
        We use Xavier initialization to set the weights of the inp2hid and hid2out connections
        """
        a = 1.0 / np.sqrt(N_INPUT)
        wt_init = np.random.uniform(low=-a, high=a, size=(N_INPUT * NUM_HIDDEN)).flatten()
        inp2hid.vars["w"].view[:] = wt_init
        model.push_var_to_device("inp2hid", "w")
        a = 1.0 / np.sqrt(NUM_HIDDEN)
        wt_init = np.random.uniform(low=-a, high=a, size=NUM_HIDDEN)
        hid2out.vars["w"].view[:] = wt_init
        model.push_var_to_device("hid2out", "w")

        if feedback_type == "random":
            feedback_wts = np.random.normal(0.0, 1.0, size=(NUM_HIDDEN, 1)).flatten()
            out2hid.vars['g'].view[:] = feedback_wts
            model.push_var_to_device('out2hid', 'g')
        elif feedback_type == "symmetric":
            model.pull_var_from_device("hid2out", "w")
            ff_wts = hid2out.vars["w"].view[:]
            ff_wts = np.transpose(np.reshape(ff_wts, newshape=(NUM_HIDDEN, 1)))
            out2hid.vars['g'].view[:] = ff_wts
            model.push_var_to_device('out2hid', 'g')
        elif feedback_type == "uniform":
            out2hid.vars['g'].view[:] = 1.0
            model.push_var_to_device('out2hid', 'g')

        spikeTimes_view = inp.extra_global_params['spikeTimes'].view
        start_spike_view = inp.vars['startSpike'].view
        out_err_tilda = out.vars['err_tilda'].view
        out_err_rise = out.vars["err_rise"].view
        out_err_decay = out.vars["err_decay"].view
        out_voltage = out.vars['V'].view
        inp_z = inp.vars['z'].view
        inp_z_tilda = inp.vars["z_tilda"].view
        hid_err_tilda = hid.vars['err_tilda'].view
        hid_z = hid.vars['z'].view
        hid_z_tilda = hid.vars['z_tilda'].view
        hid_voltage = hid.vars['V'].view
        inp2hid_lambda = inp2hid.vars['lambda'].view
        hid2out_lambda = hid2out.vars['lambda'].view
        inp2hid_e = inp2hid.vars['e'].view
        hid2out_e = hid2out.vars['e'].view

        """
        Each plot will have:
        1. Spikes of input population
        2. Membrane potential of hidden neurons
        3. Error of output neuron
        4. Target spike train
        5. Membrane potential of output neuron
        """

        print("r0 at the start: " + str(r0))

        """
        We also create all the ingredients needed to calculate the error
        """
        a = 10.0
        b = 5.0
        scale_tr_err_flt = 1.0 / ((((a * b) / (a - b)) ** 2) * (a / 2 + b / 2 - 2 * (a * b) / (a + b))) / tau_avg_err
        record_avgsqerr = np.empty(0)

        steps_in_trial = int(PRESENT_TIMESTEPS / TIME_FACTOR)

        for trial in range(TRIALS):

            # Decrease the learning rate every 600th trial
            if trial != 0 and trial % 600 == 0:
                r0 *= 0.1
                inp2hid.vars["r0"].view[:] = r0
                model.push_var_to_device('inp2hid', "r0")
                hid2out.vars["r0"].view[:] = r0
                model.push_var_to_device('hid2out', "r0")
                print("Changed r0 to: " + str(r0))

            # Reset variables at the beginning of a trial
            out_voltage[:] = OUTPUT_PARAMS["Vrest"]
            model.push_var_to_device('out', "V")
            inp_z[:] = ssa_input_init['z']
            model.push_var_to_device("inp", "z")
            inp_z_tilda[:] = ssa_input_init["z_tilda"]
            model.push_var_to_device("inp", "z_tilda")
            out_err_tilda[:] = 0.0
            model.push_var_to_device('out', 'err_tilda')
            out_err_rise[:] = 0.0
            model.push_var_to_device('out', 'err_rise')
            out_err_decay[:] = 0.0
            model.push_var_to_device('out', 'err_decay')
            hid_z[:] = hidden_init['z']
            model.push_var_to_device("hid", "z")
            hid_z_tilda[:] = hidden_init['z_tilda']
            model.push_var_to_device("hid", "z_tilda")
            hid_voltage[:] = HIDDEN_PARAMS["Vrest"]
            model.push_var_to_device("hid", "V")
            hid_err_tilda[:] = 0.0
            model.push_var_to_device('hid', 'err_tilda')

            hid2out_lambda[:] = 0.0
            model.push_var_to_device("hid2out", "lambda")
            inp2hid_lambda[:] = 0.0
            model.push_var_to_device("inp2hid", "lambda")
            hid2out_e[:] = 0.0
            model.push_var_to_device("hid2out", "e")
            inp2hid_e[:] = 0.0
            model.push_var_to_device("inp2hid", "e")

            produced_spike_train = []

            if trial % 100 == 0:
                print("Trial: " + str(trial))

            if trial != 0:
                spikeTimes += PRESENT_TIMESTEPS

                spikeTimes_view[:] = spikeTimes
                model.push_extra_global_param_to_device("inp", "spikeTimes")

                start_spike_view[:] = start_spike
                model.push_var_to_device("inp", "startSpike")

            for t in range(steps_in_trial):

                model.step_time()

                if model.t % PRESENT_TIMESTEPS == 0 and model.t != 0 and feedback_type == "symmetric":
                    model.pull_var_from_device("hid2out", "w")
                    ff_wts = hid2out.vars["w"].view[:]
                    ff_wts = np.transpose(np.reshape(ff_wts, newshape=(NUM_HIDDEN, 1)))
                    out2hid.vars['g'].view[:] = ff_wts

            # append new errors to record_avgsqerr
            model.pull_var_from_device("out", "avg_sq_err")
            avgsqrerr = out.vars["avg_sq_err"].view[:]
            error = get_mean_square_error(scale_tr_err_flt, avgsqrerr, model.t, tau_avg_err)
            record_avgsqerr = np.hstack((record_avgsqerr, error))
            out.vars["avg_sq_err"].view[:] = np.zeros(shape=1)
            model.push_var_to_device("out", "avg_sq_err")

        avgsqerr_arr[sim_idx, :] = record_avgsqerr

    fname = feedback_type + "_avgsqerr.pkl"
    with open(fname, 'wb') as f:
        pkl.dump(avgsqerr_arr, f)