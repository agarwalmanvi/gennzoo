from pygenn.genn_model import create_custom_weight_update_class

superspike_model = create_custom_weight_update_class(
    "superspike_model",
    param_names=["t_rise", "t_decay", "beta", "Vthresh"],
    var_name_types=[("w", "scalar"), ("z", "scalar"), ("z_tilda", "scalar"), ("h", "scalar"),
                    ("sigma_prime", "scalar"), ("z_der", "scalar"), ("z_tilda_der", "scalar")],
    sim_code=
    """
    $(addToInSyn, $(w));
    // presynaptic trace (3.1)
    z += ((-z / t_rise) + Isyn) * DT;
    z_tilda += ((-z_tilda + z) / t_decay) * DT;
    // surrogate partial derivative (3.2)
    h = beta * (V_post - Vthresh);
    sigma_prime = 1 / (pow(1 + abs(h), 2));
    z_der += ((-z_der / t_rise) + sigma_prime) * DT;
    z_tilda_der += ((-z_tilda_der + z_der) / t_decay) * DT;
    // output error signal
    
    """,
    post_spike_code=
    """"""
)

SUPERSPIKE_PARAMS = {"t_rise": 5,
                     "t_decay": 10,
                     "beta": 1,
                     "Vthresh": -50}