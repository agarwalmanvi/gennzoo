"""
Custom LIF neuron for SuperSpike
"""

from pygenn.genn_model import create_custom_neuron_class

lif_superspike = create_custom_neuron_class(
    "lif_superspike",
    param_names=["Tau_mem", "Vrest", "Vthresh", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("RefracTime")],
    sim_code="""
    if ($(RefracTime) <= 0.0) {
        $(Isyn) -= ($(Isyn) / $(Tsyn)) * DT;
        $(V) += (($(Vrest) - $(V) + $(Isyn)) / $(Tau_mem)) * DT;
    }
    else {
        $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(V) = $(Vrest);
    $(RefracTime) = $(TauRefrac);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)"
)

LIF_PARAMS = {"Tau_mem": 10,
              "Vrest": -60,
              "Vthresh": -50,
              "TauRefrac": 5.0}

lif_init = {"V": -60,
            "RefracTime": 0.0}

# TODO Use ExpCurr for equation 7