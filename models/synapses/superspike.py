from pygenn.genn_model import create_custom_weight_update_class

superspike_model = create_custom_weight_update_class(
    "superspike_model",
    param_names=[],
    var_name_types=[("w", "scalar"), ("e", "scalar"), ("spike_occurs", "scalar"), ("m", "scalar")],
    sim_code=
    """
    $(addToInSyn, $(w));    
    """,
    post_spike_code=
    """
    """,
    synapse_dynamics_code=
    """
    // Filtered Hebbian term
    const scalar p = z_tilda_pre * sigma_prime_post;
    p *= exp(- DT / t_rise);
    lambda += ( (- p + heb_term) / t_decay ) * DT;
    // Output error signal for precisely timed spikes
    if (sT_post == t && spike_occurs == False) {
        const scalar mismatch = -1.0
    } else if (sT_post != t && spike_occurs == True) {
        const scalar mismatch = 1.0
    } else {
        const scalar mismatch = 0.0
    }
    error = alpha * mismatch;
    // insert code to calculate learning rate r
    // at each time step, calculate m
    if (t % 500) {
        w += r * m;
        w = fmin(wmax, fmax(wmin, w));
        m = 0.0;
    }
    m += lambda * error;
    """,
    is_post_spike_time_required=True
)

SUPERSPIKE_PARAMS = {}