"""
Custom LIF neuron for SuperSpike
"""

from pygenn.genn_model import create_custom_neuron_class, create_dpf_class, init_var
from numpy import exp, log, random

# OUTPUT NEURON MODEL for precisely timed spikes #
output_model = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta", "t_peak"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar"),
                    ("err_rise", "scalar"), ("err_tilda", "scalar"), ("err_decay", "scalar"),
                    ("mismatch", "scalar")],
    sim_code="""
    // membrane potential dynamics
    if ($(RefracTime) == $(TauRefrac)) {
        $(V) = $(Vrest);
    }
    if ($(RefracTime) <= 0.0) {
        scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(RefracTime) -= DT;
    }
    // filtered partial derivative
    const scalar one_plus_hi = 1.0 + fabs($(beta) * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    // error
    const scalar S_pred = $(spike_times)[(int)round($(t) / DT)];
    const scalar S_real = $(RefracTime) <= 0.0 && $(V) >= $(Vthresh) ? 1.0 : 0.0;
    $(mismatch) = S_pred - S_real;
    $(err_rise) = ($(err_rise) * $(t_rise_mult)) + $(mismatch);
    $(err_decay) = ($(err_decay) * $(t_decay_mult)) + $(mismatch);
    $(err_tilda) = ($(err_decay) - $(err_rise)) * $(norm_factor);
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
        ("norm_factor", create_dpf_class(lambda pars, dt:
                                         1.0 / (- exp(- pars[9] / pars[6]) + exp(
                                             - pars[9] / pars[7])))()),
        ("t_rise_mult", create_dpf_class(lambda pars, dt: exp(- dt / pars[6]))()),
        ("t_decay_mult", create_dpf_class(lambda pars, dt: exp(- dt / pars[7]))())
    ],
    extra_global_params=[("spike_times", "scalar*")]
)

## spike_times should be a binary array that captures the target binary spike train

OUTPUT_PARAMS = {"C": 10.0,
                 "Tau_mem": 10.0,
                 "Vrest": -60.0,
                 "Vthresh": -50.0,
                 "Ioffset": 0.0,
                 "TauRefrac": 5.0,
                 "t_rise": 5.0,
                 "t_decay": 10.0,
                 "beta": 1.0}

OUTPUT_PARAMS["t_peak"] = ((OUTPUT_PARAMS["t_decay"] * OUTPUT_PARAMS["t_rise"]) / (
        OUTPUT_PARAMS["t_decay"] - OUTPUT_PARAMS["t_rise"])) \
                          * log(OUTPUT_PARAMS["t_decay"] / OUTPUT_PARAMS["t_rise"])

output_init = {"V": -60,
               "RefracTime": 0.0,
               "sigma_prime": 0.0,
               "err_rise": 0.0,
               "err_decay": 0.0,
               "err_tilda": 0.0,
               "mismatch": 0.0}

# HIDDEN NEURON MODEL #

NUM_HIDDEN = 8

hidden_model = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac", "beta", "t_rise", "t_decay"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar"),
                    ("err_tilda", "scalar"), ("feedback_mult", "scalar"), ("z", "scalar"),
                    ("z_tilda", "scalar"), ("err_output", "scalar")],
    sim_code="""
    // membrane potential dynamics
    if ($(RefracTime) == $(TauRefrac)) {
        $(V) = $(Vrest);
    }
    if ($(RefracTime) <= 0.0) {
        scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(RefracTime) -= DT;
    }
    // filtered presynaptic trace
    $(z) += (- $(z) / $(t_rise)) * DT;
    $(z_tilda) += ((- $(z_tilda) + $(z)) / $(t_decay)) * DT;
    if ($(z_tilda) < 0.0000001) {
        $(z_tilda) = 0.0;
    }
    // filtered partial derivative
    const scalar one_plus_hi = 1.0 + fabs($(beta) * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    // error
    $(err_tilda) = $(err_output) * $(feedback_mult);
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(z) += 1.0;
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])())
    ]
)

HIDDEN_PARAMS = {"C": 10.0,
                 "Tau_mem": 10.0,
                 "Vrest": -60.0,
                 "Vthresh": -50.0,
                 "Ioffset": 0.0,
                 "TauRefrac": 5.0,
                 "beta": 1.0,
                 "t_rise": 5.0,
                 "t_decay": 10.0}

hidden_init = {"V": -60,
               "RefracTime": 0.0,
               "sigma_prime": 0.0,
               "err_tilda": 0.0,
               "feedback_mult": init_var("Normal", {"mean": 0.0, "sd": 1.0}),
               "z": 0.0,
               "z_tilda": 0.0,
               "err_output": 0.0}

# OUTPUT NEURON MODEL for classifying patterns #

output_model_classification = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta", "t_peak"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar"),
                    ("err_rise", "scalar"), ("err_tilda", "scalar"), ("err_decay", "scalar"),
                    ("mismatch", "scalar"),
                    ("S_pred", "scalar"), ("S_miss", "scalar"), ("window_of_opp", "scalar")],
    sim_code="""
    // membrane potential dynamics
    if ($(RefracTime) == $(TauRefrac)) {
        $(V) = $(Vrest);
    }
    if ($(RefracTime) <= 0.0) {
        scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(RefracTime) -= DT;
    }
    // filtered partial derivative
    const scalar one_plus_hi = 1.0 + fabs($(beta) * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    // error
    $(mismatch) = 0.0;
    if ($(window_of_opp) == 1.0) {
        if ($(S_pred) == 0.0) {
            const scalar S_real = $(RefracTime) <= 0.0 && $(V) >= $(Vthresh) ? 1.0 : 0.0;
            if (S_real == 1.0) {
                $(mismatch) = -1.0;
            }
        }
    }
    else {
        if ($(S_pred) == 1.0 && $(S_miss) == 1.0) {
            $(mismatch) = 1.0;
        }
    }
    $(err_rise) = ($(err_rise) * $(t_rise_mult)) + $(mismatch);
    $(err_decay) = ($(err_decay) * $(t_decay_mult)) + $(mismatch);
    $(err_tilda) = ($(err_decay) - $(err_rise)) * $(norm_factor);
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
        ("norm_factor", create_dpf_class(lambda pars, dt:
                                         1.0 / (- exp(- pars[9] / pars[6]) + exp(
                                             - pars[9] / pars[7])))()),
        ("t_rise_mult", create_dpf_class(lambda pars, dt: exp(- dt / pars[6]))()),
        ("t_decay_mult", create_dpf_class(lambda pars, dt: exp(- dt / pars[7]))())
    ]
)

output_init_classification = {"V": -60,
                              "RefracTime": 0.0,
                              "sigma_prime": 0.0,
                              "err_rise": 0.0,
                              "err_decay": 0.0,
                              "err_tilda": 0.0,
                              "mismatch": 0.0,
                              "S_pred": 0.0,
                              "S_miss": 0.0,
                              "window_of_opp": 0.0}

# S_pred is 0/1 indicating if this is the target neuron -- should be considered only during window of opportunity
# S_miss is 0/1 to indicate if this neuron should have fired during the window of opportunity and did not
#                           -- should only be considered for one time step at the end of the window of opportunity
# window_of_opp is 0/1 to indicate if this timestep falls within the window of opportunity when the output neuron
# is expected to react
