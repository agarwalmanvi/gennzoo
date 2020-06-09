"""
Leaky Integrate-and-fire neuron
Parameters:
Variables:

Neuron model from:
F. Zenke and S. Ganguli,
"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks,"
in Neural Computation, vol. 30, no. 6, pp. 1514-1541, 2018,
doi: 10.1162/neco_a_01086
"""

from pygenn.genn_model import create_custom_neuron_class

lif_model = create_custom_neuron_class(
    "lif_model",
    param_names=[],
    var_name_types=[],
    sim_code="""
    """,
    reset_code="""
    """,
    threshold_condition_code=""
)

LIF_PARAMS = {}

lif_init = {}