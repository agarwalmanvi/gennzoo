from pygenn.genn_model import create_custom_weight_update_class, create_dpf_class, init_var
from numpy import exp

superspike_model = create_custom_weight_update_class(
    "superspike_model",
    param_names=["t_rise", "t_decay", "tau_rms", "r0", "wmax", "wmin", "epsilon"],
    var_name_types=[("w", "scalar"), ("e", "scalar"), ("lambda", "scalar"), ("upsilon", "scalar"),
                    ("m", "scalar"), ("trial_length", "scalar"), ("ExpRMS", "scalar"),
                    ("trial_end_t", "scalar")],
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
    if ((int)round($(t)) % (int)$(trial_end_t) == 0 && (int)round($(t)) != 0) {
        // calculate learning rate r
        $(ExpRMS) = exp( - $(trial_length) / $(tau_rms));
        const scalar grad= $(m)/$(trial_length);
        $(upsilon) = fmax($(upsilon) * $(ExpRMS) , grad*grad);
        //$(upsilon) = 1.0;
        const scalar r = $(r0) / (sqrt($(upsilon))+$(epsilon));
        $(w) += r * grad;
        $(w) = fmin($(wmax), fmax($(wmin), $(w)));
        $(m) = 0.0;
    }
    """,
    derived_params=[
        ("ExpRMS_old", create_dpf_class(lambda pars, dt: exp(- 500.0 / pars[2]))())
    ]
)

SUPERSPIKE_PARAMS = {"t_rise": 5,
                     "t_decay": 10,
                     "tau_rms": 300,
                     "r0": 0.01,
                     "wmax": 10.0,
                     "wmin": -10.0,
                     "epsilon": 0.000000000000000000001}

superspike_init = {"w": init_var("Uniform", {"min": -0.0001, "max": 0.0001}),
                   "e": 0.0,
                   "lambda": 0.0,
                   "upsilon": 0.0,
                   "m": 0.0,
                   "trial_length": 0.0,
                   "ExpRMS": 0.0,
                   "trial_end_t": 0.0}
