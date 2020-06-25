from pygenn.genn_model import create_custom_weight_update_class, create_dpf_class, init_var
from numpy import exp

superspike_model = create_custom_weight_update_class(
    "superspike_model",
    param_names=["t_rise", "t_decay", "tau_rms", "r0", "wmax", "wmin"],
    var_name_types=[("w", "scalar"), ("e", "scalar"), ("lambda", "scalar"), ("upsilon", "scalar"), ("m", "scalar")],
    synapse_dynamics_code=
    """
    $(addToInSyn, $(w));
    // Filtered eligibility trace
    $(e) += $(z_tilda_pre) * $(sigma_prime_post);
    $(e) *= exp(- DT / $(t_rise));
    $(lambda) += ( (- $(lambda) + $(e)) / $(t_decay)) * DT;
    // get error from neuron model and compute full expression under integral
    const scalar g = $(lambda) * $(err_tilda_post);
    // calculate learning rate r
    $(upsilon) = fmax($(upsilon) * $(ExpRMS) , g * g);
    // at each time step, calculate m
    $(m) += g;
    if ((int)round($(t)) % 500 == 0 && (int)round($(t)) != 0) {
        const scalar r = $(r0) / sqrt($(upsilon));
        $(w) += r * $(m);
        $(w) = fmin($(wmax), fmax($(wmin), $(w)));
        $(m) = 0.0;
    }
    """,
    derived_params=[
        ("ExpRMS", create_dpf_class(lambda pars, dt: exp(- dt / pars[2]))())
    ]
)

SUPERSPIKE_PARAMS = {"t_rise": 5,
                     "t_decay": 10,
                     "tau_rms": 10,
                     "r0": 1.0,
                     "wmax": 0.1,
                     "wmin": -0.1}

superspike_init = {"w": init_var("Uniform", {"min": -0.001, "max": 0.001}),
                   "e": 0.0,
                   "lambda": 0.0,
                   "upsilon": 0.0,
                   "m": 0.0}