from pygenn.genn_model import create_custom_neuron_class

if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vtheta", "lambda", "Vrest", "Vreset"],
    var_name_types=[("V", "scalar")],
    sim_code="""
    if ($(V) >= $(Vtheta)) {
        $(V) = $(Vreset);
    }
    $(V) += (-$(lambda) + $(Isyn)) * DT;
    $(V) = fmax($(V), $(Vrest));
    """,
    reset_code="""
    """,
    threshold_condition_code="$(V) >= $(Vtheta)"
)

IF_PARAMS = {"Vtheta": 1.0,
             "lambda": 0.01,
             "Vrest": 0.0,
             "Vreset": 0.0}

if_init = {"V": 0.0}