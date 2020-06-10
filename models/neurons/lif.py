"""
Leaky Integrate-and-fire neuron
This is an example of what parameters and variables you need to specify
in order to initialize the in-built LIF model class.
This model is used with the SuperSpike learning rule.
Parameters: C -> Membrane capacitance
            TauM -> Membrane time constant
            Vrest -> Resting membrane potential
            Vreset -> Reset voltage
            Vthresh -> Spiking threshold
            Ioffset -> Offset current
            TauRefrac -> Length of refractory period
Variables: V -> Membrane potential
           RefracTime -> Counter to check if neuron is in refractory period

Neuron model from:
F. Zenke and S. Ganguli,
"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks,"
in Neural Computation, vol. 30, no. 6, pp. 1514-1541, 2018,
doi: 10.1162/neco_a_01086
"""

LIF_PARAMS = {"C": 1.0,
              "TauM": 10,
              "Vrest": -60,
              "Vreset": -60,
              "Vthresh": -50,
              "Ioffset": 0.0,
              "TauRefrac": 5.0}

lif_init = {"V": -60,
            "RefracTime": 0.0}