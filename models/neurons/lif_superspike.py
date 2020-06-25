"""
Custom LIF neuron for SuperSpike
"""

from pygenn.genn_model import create_custom_neuron_class, create_dpf_class
from numpy import exp, log

lif_model = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta", "t_peak"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("z", "scalar"), ("z_tilda", "scalar"),
                    ("sigma_prime", "scalar"), ("err", "scalar"), ("err_tilda", "scalar")],
    sim_code="""
    // membrane potential dynamics
    if ($(RefracTime) <= 0.0) {
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
    // error
    const scalar S_pred = $(spike_times)[(int)round($(t) / DT)];
    const scalar S_real = $(RefracTime) <= 0.0 && $(V) >= $(Vthresh) ? 1.0 : 0.0;
    const scalar mismatch = S_pred - S_real;
    $(err) += ( (- $(err) / $(t_rise)) + mismatch ) * DT;
    $(err_tilda) = ( ( - $(err_tilda) + $(err) ) / $(t_decay) ) * DT;
    // normalize to unity to give final error - take approach 1
    // const scalar norm_factor = 1.0 / (- exp(- $(t_peak) / $(t_rise)) + exp(- $(t_peak) / $(t_decay)));
    $(err_tilda) = $(err_tilda) / $(norm_factor);
    """,
    reset_code="""
    $(V) = $(Vrest);
    $(RefracTime) = $(TauRefrac);
    $(z) += 1.0;
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
        ("norm_factor", create_dpf_class(lambda pars, dt:
                                         1.0 / (- exp(- pars[9] / pars[6]) + exp(- pars[9] / pars[7])))())
    ],
    extra_global_params=[("spike_times", "scalar*")]
)

lif_hidden_model = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta", "t_peak"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("z", "scalar"), ("z_tilda", "scalar"),
                    ("sigma_prime", "scalar"), ("err", "scalar"), ("err_tilda", "scalar")],
    sim_code="""
    // membrane potential dynamics
    if ($(RefracTime) <= 0.0) {
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
    // error
    const scalar S_pred = $(spike_times)[(int)round($(t) / DT)];
    const scalar S_real = $(RefracTime) <= 0.0 && $(V) >= $(Vthresh) ? 1.0 : 0.0;
    const scalar mismatch = S_pred - S_real;
    $(err) += ( (- $(err) / $(t_rise)) + mismatch ) * DT;
    $(err_tilda) = ( ( - $(err_tilda) + $(err) ) / $(t_decay) ) * DT;
    // normalize to unity to give final error - take approach 1
    // const scalar norm_factor = 1.0 / (- exp(- $(t_peak) / $(t_rise)) + exp(- $(t_peak) / $(t_decay)));
    $(err_tilda) = $(err_tilda) / $(norm_factor);
    """,
    reset_code="""
    $(V) = $(Vrest);
    $(RefracTime) = $(TauRefrac);
    $(z) += 1.0;
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
        ("norm_factor", create_dpf_class(lambda pars, dt:
                                         1.0 / (- exp(- pars[9] / pars[6]) + exp(- pars[9] / pars[7])))())
    ],
    extra_global_params=[("spike_times", "scalar*")]
)

## spike_times should be a binary array that captures the target binary spike train

LIF_PARAMS = {"C": 1.0,
              "Tau_mem": 10.0,
              "Vrest": -60.0,
              "Vthresh": -50.0,
              "Ioffset": 0.0,
              "TauRefrac": 5.0,
              "t_rise": 5.0,
              "t_decay": 10.0,
              "beta": 1.0}

LIF_PARAMS["t_peak"] = ( (LIF_PARAMS["t_decay"] * LIF_PARAMS["t_rise"]) / (LIF_PARAMS["t_decay"] - LIF_PARAMS["t_rise"]) ) * log(LIF_PARAMS["t_decay"] / LIF_PARAMS["t_rise"])

lif_init = {"V": -60,
            "RefracTime": 0.0,
            "z": 0.0,
            "z_tilda": 0.0,
            "sigma_prime": 0.0,
            "err": 0.0,
            "err_tilda": 0.0}