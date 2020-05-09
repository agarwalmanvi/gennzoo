# Sample script for training the weight update model from
# "Learning real world stimuli in a neural network with spike-driven synaptic dynamics"
# by Brader, Senn & Fusi

from models.neurons.integrate_fire import if_model, IF_PARAMS, if_init
from models.synapses.fusi import fusi_model, FUSI_PARAMS, fusi_init, fusi_post_init

print("Imported")

