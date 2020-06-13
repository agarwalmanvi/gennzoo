from pygenn.genn_model import create_custom_weight_update_class

superspike_model = create_custom_weight_update_class(
    "superspike_model",
    param_names=["t_rise", "t_decay", "tau_rms", "r0", "wmax", "wmin"],
    var_name_types=[("w", "scalar"), ("e", "scalar"), ("lambda", "scalar"), ("upsilon", "scalar"), ("m", "scalar")],
    synapse_dynamics_code=
    """
    $(addToInSyn, $(w));
    // Filtered Hebbian term
    $(e) += $(z_tilda_pre) * $(sigma_prime_post);
    $(e) *= exp(- DT / $(t_rise));
    $(lambda) += ( (- $(lambda) + $(e)) / $(t_decay)) * DT;
    // get error from neuron model
    const scalar g = $(lambda) * $(e_post);
    // calculate learning rate r
    $(upsilon) = fmax($(upsilon) * exp( - DT / $(tau_rms)) , g * g);
    const scalar r = $(r0) / sqrt($(upsilon));
    // at each time step, calculate m
    if ($(t) % 500 == 0) {
        $(w) += r * $(m);
        $(w) = fmin($(wmax), fmax($(wmin), $(w)));
        $(m) = 0.0;
    }
    $(m) += g;
    """
)

SUPERSPIKE_PARAMS = {}