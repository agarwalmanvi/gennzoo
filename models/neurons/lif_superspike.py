from pygenn.genn_model import create_custom_neuron_class, create_dpf_class, init_var, create_custom_postsynaptic_class
from numpy import exp, log, random
from models.parameters import *


"""
Output neuron model for task to reproduce precisely timed spikes
This is a general model for the case of multiple output neurons.
"""
output_model = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta", "t_peak", "tau_avg_err"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar"),
                    ("err_rise", "scalar"), ("err_tilda", "scalar"), ("err_decay", "scalar"),
                    ("startSpike", "unsigned int"), ("endSpike", "unsigned int"),
                    ("avg_sq_err", "scalar")],
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
    const scalar one_plus_hi = 1.0 + fabs($(beta) * 0.001 * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    // error
    scalar S_pred = 0.0;
    if ($(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]) {
        $(startSpike)++;
        S_pred = 1.0;
    }
    const scalar S_real = $(RefracTime) <= 0.0 && $(V) >= $(Vthresh) ? 1.0 : 0.0;
    const scalar mismatch = S_pred - S_real;
    $(err_rise) = ($(err_rise) * $(t_rise_mult)) + mismatch;
    $(err_decay) = ($(err_decay) * $(t_decay_mult)) + mismatch;
    $(err_tilda) = ($(err_decay) - $(err_rise)) * $(norm_factor);
    // calculate average error trace
    const scalar temp = $(err_tilda) * $(err_tilda) * DT * 0.001;
    $(avg_sq_err) *= $(mul_avgerr);
    $(avg_sq_err) += temp;
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
        ("t_decay_mult", create_dpf_class(lambda pars, dt: exp(- dt / pars[7]))()),
        ("mul_avgerr", create_dpf_class(lambda pars, dt: exp(-dt / pars[10]))())
    ],
    extra_global_params=[("spikeTimes", "scalar*")]
)

OUTPUT_PARAMS = {"C": 10.0,
                 "Tau_mem": 10.0,
                 "Vrest": -60.0,
                 "Vthresh": -50.0,
                 "Ioffset": 0.0,
                 "TauRefrac": 5.0,
                 "t_rise": t_rise,
                 "t_decay": t_decay,
                 "beta": 1.0,
                 "tau_avg_err": tau_avg_err}

OUTPUT_PARAMS["t_peak"] = ((OUTPUT_PARAMS["t_decay"] * OUTPUT_PARAMS["t_rise"]) / (
        OUTPUT_PARAMS["t_decay"] - OUTPUT_PARAMS["t_rise"])) \
                          * log(OUTPUT_PARAMS["t_decay"] / OUTPUT_PARAMS["t_rise"])

output_init = {"V": -60,
               "RefracTime": 0.0,
               "sigma_prime": 0.0,
               "err_rise": 0.0,
               "err_decay": 0.0,
               "err_tilda": 0.0,
               "avg_sq_err": 0.0}
# startSpike and endSpike to be specified in simulation script


""" Neuron model for hidden layers """

hidden_model = create_custom_neuron_class(
   "lif_superspike",
   param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac", "beta", "t_rise", "t_decay"],
   var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar"),
                   ("err_tilda", "scalar"), ("z", "scalar"), ("z_tilda", "scalar")],
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
    const scalar one_plus_hi = 1.0 + fabs($(beta) * 0.001 * ($(V) - $(Vthresh)));
    $(sigma_prime) = 1.0 / (one_plus_hi * one_plus_hi);
    // error
    $(err_tilda) = $(ISynFeedback);
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(z) += 1.0;
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", create_dpf_class(lambda pars, dt: exp(-dt / pars[1]))()),
        ("Rmembrane", create_dpf_class(lambda pars, dt: pars[1] / pars[0])())
    ],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)]
)

HIDDEN_PARAMS = {"C": 10.0,
                 "Tau_mem": 10.0,
                 "Vrest": -60.0,
                 "Vthresh": -50.0,
                 "Ioffset": 0.0,
                 "TauRefrac": 5.0,
                 "beta": 1.0,
                 "t_rise": t_rise,
                 "t_decay": t_decay}

hidden_init = {"V": -60,
               "RefracTime": 0.0,
               "sigma_prime": 0.0,
               "err_tilda": 0.0,
               "z": 0.0,
               "z_tilda": 0.0}


""" Output neuron model for classification """

output_model_classification = create_custom_neuron_class(
    "lif_superspike",
    param_names=["C", "Tau_mem", "Vrest", "Vthresh", "Ioffset", "TauRefrac",
                 "t_rise", "t_decay", "beta", "t_peak", "tau_avg_err"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("sigma_prime", "scalar"),
                    ("err_rise", "scalar"), ("err_tilda", "scalar"), ("err_decay", "scalar"),
                    ("mismatch", "scalar"),
                    ("S_pred", "scalar"), ("S_miss", "scalar"), ("window_of_opp", "scalar"),
                   ("avg_sq_err", "scalar")],
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
    const scalar one_plus_hi = 1.0 + fabs($(beta) * 0.001 * ($(V) - $(Vthresh)));
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
    // calculate average error trace
    const scalar temp = $(err_tilda) * $(err_tilda) * DT * 0.001;
    $(avg_sq_err) *= $(mul_avgerr);
    $(avg_sq_err) += temp;
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
        ("t_decay_mult", create_dpf_class(lambda pars, dt: exp(- dt / pars[7]))()),
        ("mul_avgerr", create_dpf_class(lambda pars, dt: exp(-dt / pars[10]))())
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
                              "window_of_opp": 0.0,
                               "avg_sq_err": 0.0}

# Since this neuron model has the same parameters as `output_model`, we can reuse OUTPUT_PARAMS here.

"""
Some notes on how to interpret some of variables in this model:
1. S_pred is 0/1 indicating if this is the target neuron -- should be considered only during window of opportunity
2. S_miss is 0/1 to indicate if this neuron should have fired during the window of opportunity and did not
                          -- should only be considered for one time step at the end of the window of opportunity
3. window_of_opp is 0/1 to indicate if this timestep falls within the window of opportunity when the output neuron
is expected to react
"""

""" Postsynaptic model to transfer input directly into a neuron without any dynamics """

feedback_postsyn_model = create_custom_postsynaptic_class(
    "Feedback",
    apply_input_code="$(ISynFeedback) += $(inSyn); $(inSyn) = 0;")
