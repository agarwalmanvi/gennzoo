"""
Custom LIF neuron for SuperSpike
"""

from pygenn.genn_model import create_custom_neuron_class, create_dpf_class
from numpy import exp

lif_superspike = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac", "t_rise", "t_decay", "beta"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("z", "scalar"), ("z_tilda", "scalar"),
                    ("sigma_prime", "scalar"), ("z_der", "scalar"), ("z_tilda_der", "scalar")],
    sim_code="""
    if ($(RefracTime) <= 0.0) {
        // $(V) += (($(Vrest) - $(V) + $(Isyn)) / $(Tau_mem)) * DT;
        scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(RefracTime) -= DT;
    }
    // filtered presynaptic trace
    $(z) *= exp(- DT / $(t_rise));
    $(z_tilda) += ((- $(z_tilda) + $(z)) / $(t_decay)) * DT;
    // filtered partial derivative
    $(sigma_prime) = 1 / (pow(1 + abs(($(beta) * ($(V) - $(Vthresh)))), 2));
    $(z_der) += ((- $(z_der) / $(t_rise)) + $(sigma_prime)) * DT;
    $(z_tilda_der) += ((- $(z_tilda_der) + $(z_der)) / $(t_decay)) * DT;
    """,
    reset_code="""
    $(V) = $(Vrest);
    $(RefracTime) = $(TauRefrac);
    $(z) += 1.0;
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", genn_model.create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])())
    ]
)

LIF_PARAMS = {"C": 1.0,
              "TauM": 10,
              "Vrest": -60,
              "Vthresh": -50,
              "Ioffset": 0.0,
              "TauRefrac": 5.0,
              "t_rise": 5,
              "t_decay": 10,
              "beta": 1.0}

lif_init = {"V": -60,
            "RefracTime": 0.0,
            "z": 0.0,
            "z_tilda": 0.0,
            "sigma_prime": 0.0,
            "z_der": 0.0,
            "z_tilda_der": 0.0}

# TODO Use ExpCurr for equation 3.2