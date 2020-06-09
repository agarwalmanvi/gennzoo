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

# TODO refractory period
lif_model = create_custom_neuron_class(
    "lif_model",
    param_names=["Urest", "Tmem", "Tsyn", "Uthr"],
    var_name_types=["U"],
    sim_code="""
    $(U) += (($(Urest) - $(U) + $(Isyn)) / $(Tmem)) * DT;
    $(Isyn) -= ($(Isyn) / $(Tsyn)) * DT; // This seems wrong
    """,
    reset_code="""
    $(U) = $(Urest);
    """,
    threshold_condition_code="$(U) >= $(Uthr)"
)

LIF_PARAMS = {"Urest": -60,
              "Tmem": 10,
              "Tsyn": 5,
              "Uthr": -50}

lif_init = {"U": -60}