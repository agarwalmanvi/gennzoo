from pygenn.genn_model import create_custom_weight_update_class, create_dpf_class, init_var
from numpy import exp
from models.parameters import *

""" Weight update model for feedforward weights in SuperSpike """
superspike_model = create_custom_weight_update_class(
    "superspike_model",
    param_names=["t_rise", "t_decay", "tau_rms", "wmax", "wmin", "epsilon", "update_t"],
    var_name_types=[("w", "scalar"), ("e", "scalar"), ("lambda", "scalar"), ("upsilon", "scalar"),
                    ("m", "scalar"), ("r0", "scalar")],
    sim_code="""
    $(addToInSyn, $(w));
    """,
    synapse_dynamics_code=
    """
    // Filtered eligibility trace
    $(e) += ($(z_tilda_pre) * $(sigma_prime_post) - $(e)/$(t_rise))*DT;
    $(lambda) += ((- $(lambda) + $(e)) / $(t_decay)) * DT;
    // get error from neuron model and compute full expression under integral
    const scalar g = $(lambda) * $(err_tilda_post);
    // at each time step, calculate m
    $(m) += g;
    if ((int)round($(t)) % (int)$(update_t) == 0 && (int)round($(t)) != 0) {
        const scalar grad = $(m)/$(update_t);
        // calculate learning rate r
        $(upsilon) = fmax($(upsilon) * $(ExpRMS) , grad*grad);
        const scalar r = $(r0) / (sqrt($(upsilon))+$(epsilon));
        // update synaptic weight
        $(w) += r * grad;
        $(w) = fmin($(wmax), fmax($(wmin), $(w)));
        $(m) = 0.0;
    }
    """,
    derived_params=[
        ("ExpRMS", create_dpf_class(lambda pars, dt: exp(- pars[-1] / pars[2]))())
    ]
)

SUPERSPIKE_PARAMS = {"t_rise": t_rise,
                     "t_decay": t_decay,
                     "tau_rms": tau_rms,
                     "wmax": wmax,
                     "wmin": wmin,
                     "epsilon": epsilon,
                     "update_t": update_time_ms}

superspike_init = {"w": init_var("Uniform", {"min": -0.001, "max": 0.001}),
                   "e": 0.0,
                   "lambda": 0.0,
                   "upsilon": 0.0,
                   "m": 0.0,
                   "r0": r0}

""" Feedback weights model for out2hid connections """

feedback_wts_model = create_custom_weight_update_class(
    "feedback",
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(err_tilda_pre));")

feedback_wts_init = {"g": init_var("Uniform", {"min": -0.001, "max": 0.001})}

# TODO fix regularized superspike model
# #### Superspike with regularizer ##########
#
# superspike_reg_model = create_custom_weight_update_class(
#     "superspike_model",
#     param_names=["t_rise", "t_decay", "tau_rms", "r0", "wmax", "wmin", "epsilon", "tau_het", "rho", "a,
#     var_name_types=[("w", "scalar"), ("e", "scalar"), ("lambda", "scalar"), ("upsilon", "scalar"),
#                     ("m", "scalar"), ("trial_length", "scalar"), ("ExpRMS", "scalar"),
#                     ("trial_end_t", "scalar"), ("z_fir_rate", "scalar")],
#     sim_code="""
#     $(addToInSyn, $(w));
#     """,
#     synapse_dynamics_code=
#     """
#     // Filtered eligibility trace
#     $(e) += ($(z_tilda_pre) * $(sigma_prime_post) - $(e)/$(t_rise))*DT;
#     $(lambda) += ((- $(lambda) + $(e)) / $(t_decay)) * DT;
#     // calculate regularizer
#     $(z_fir_rate) += (- $(z_fir_rate) / $(tau_het)) + $(did_i_spike_post);
#     const scalar zeta = pow($(z_fir_rate), $(a));
#     const scalar reg = $(rho) * $(w) * zeta;
#     // get error from neuron model and compute full expression under integral
#     const scalar g = $(lambda) * $(err_tilda_post);
#     // at each time step, calculate m
#     $(m) += g - reg;
#     if ((int)round($(t)) % (int)$(trial_end_t) == 0 && (int)round($(t)) != 0) {
#         // calculate learning rate r
#         $(ExpRMS) = exp( - $(trial_length) / $(tau_rms));
#         const scalar grad = $(m)/$(trial_length);
#         $(upsilon) = fmax($(upsilon) * $(ExpRMS) , grad*grad);
#         const scalar r = $(r0) / (sqrt($(upsilon))+$(epsilon));
#         // update synaptic weight
#         $(w) += r * grad;
#         $(w) = fmin($(wmax), fmax($(wmin), $(w)));
#         $(m) = 0.0;
#     }
#     """
# )
#
# SUPERSPIKE_REG_PARAMS = {"t_rise": 5,
#                          "t_decay": 10,
#                          "tau_rms": 30000,
#                          "r0": 0.01,
#                          "wmax": 10,
#                          "wmin": -10,
#                          "epsilon": 0.000000000000000000001,
#                          "tau_het": 100,
#                          "rho": 1.0,
#                          "a": 4.0}
#
# superspike_reg_init = {"w": init_var("Uniform", {"min": -0.001, "max": 0.001}),
#                        "e": 0.0,
#                        "lambda": 0.0,
#                        "upsilon": 0.0,
#                        "m": 0.0,
#                        "trial_length": 0.0,
#                        "ExpRMS": 0.0,
#                        "trial_end_t": 500.0,
#                        "z_fir_rate": 0.0}
