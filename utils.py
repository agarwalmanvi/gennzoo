import numpy as np


def create_poisson_spikes(interval, freq, spike_dt, time_factor):
    """
	Create poisson spike train for 1 neuron
	:param interval: time period to create spikes for (ms)
	:param freq: spiking frequency (Hz)
	:param spike_dt: length of smallest timestep (if 1 ms, use 0.001)
	:return: spike times in ms (np.array) within given time interval for 1 neuron
	"""
    compare_num = freq * (spike_dt * time_factor)
    spike_train = np.random.random_sample(int(interval / time_factor))
    spike_train = (spike_train < compare_num).astype(int)
    spike_times_gen = np.nonzero(spike_train)[0]
    spike_times_gen = np.multiply(spike_times_gen, time_factor)
    return spike_times_gen


def get_mean_square_error(scale, avgsqrerr, get_time, tau):
    """
    Calculate the error for the learning curve
    :param scale: scaling factor for calculating the error
    :param avgsqrerr: array of errors from each output neuron
    :param get_time: time elapsed since the start of the simulation (ms)
    :param tau: time constant of weighting the contribution of errors
    :return: mean square error at the end of one update interval
    """
    temp = scale * np.mean(avgsqrerr)
    time_in_secs = get_time / 1000
    div = 1.0 - np.exp(-time_in_secs / tau) + 1e-9
    error = temp / div
    return error
