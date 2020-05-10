from pygenn.genn_model import create_custom_weight_update_class

fusi_model = create_custom_weight_update_class(
    "fusi_model",
    param_names=["tauC", "a", "b", "thetaV", "thetaLUp", "thetaLDown", "thetaHUp", "thetaHDown",
                 "thetaX", "alpha", "beta", "Xmax", "Xmin", "JC", "Jplus", "Jminus"],
    var_name_types=[("X", "scalar"), ("last_tpre", "scalar"), ("g", "scalar")],
    post_var_name_types=[("C", "scalar")],
    sim_code="""
    $(addToInSyn, $(g));
    const scalar dt = $(t) - $(sT_post);
    const scalar decayC = $(C) * exp(-dt / $(tauC));
    if ($(V_post) > $(thetaV) && $(thetaLUp) < decayC && decayC < $(thetaHUp)) {
        $(X) += $(a);
    }
    else if ($(V_post) <= $(thetaV) && $(thetaLDown) < decayC && decayC < $(thetaHDown)) {
        $(X) -= $(b);
    }
    else {
        const scalar X_dt = $(t) - $(last_tpre);
        if ($(X) > $(thetaX)) {
            $(X) += $(alpha) * X_dt;
        }
        else {
            $(X) -= $(beta) * X_dt;
        }
    }
    $(X) = fmin($(Xmax), fmax($(Xmin), $(X)));
    $(g) = ($(X) > $(thetaX)) ? $(Jplus) : $(Jminus);
    $(last_tpre) = $(t);
    """,
    post_spike_code="""
    const scalar dt = $(t) - $(sT_post);
    $(C) = ($(C) * exp(-dt / $(tauC))) + $(JC);
    """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True
)

FUSI_PARAMS = {"tauC": 60.0, "a": 0.1, "b": 0.1, "thetaV": 0.8, "thetaLUp": 3.0,
               "thetaLDown": 3.0, "thetaHUp": 13.0, "thetaHDown": 4.0, "thetaX": 0.5,
               "alpha": 0.0035, "beta": 0.0035, "Xmax": 1.0, "Xmin": 0.0, "JC": 1.0,
               "Jplus": 1.0, "Jminus": 0.0}

fusi_init = {"X": 0.0,
             "last_tpre": 0.0,
             "g": 0.0}
fusi_post_init = {"C": 2.0}
