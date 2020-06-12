"""
Custom LIF neuron for SuperSpike
"""

from pygenn.genn_model import create_custom_neuron_class, create_dpf_class
from numpy import exp

lif_model = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac", "t_rise", "t_decay", "beta"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("z", "scalar"), ("z_tilda", "scalar"),
                    ("sigma_prime", "scalar")],
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
    const scalar one_plus_hi = 1.0 + fabs($(beta) * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    if ($(t) == 71) {
    }
        
    }
    """,
    reset_code="""
    $(V) = $(Vrest);
    $(RefracTime) = $(TauRefrac);
    $(z) += 1.0;
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])())
    ]
)

LIF_PARAMS = {"C": 1.0,
              "Tau_mem": 10,
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
            "sigma_prime": 0.0}

# TODO Use ExpCurr for equation 3.2