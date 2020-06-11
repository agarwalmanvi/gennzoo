from pygenn.genn_model import create_custom_neuron_class

poisson_const = create_custom_neuron_class(
    "poisson_const",
    param_names=["spike_train"],
    var_name_types=[("counter", "scalar")],
    sim_code="""
    counter += 1;
    """,
    threshold_condition_code="$(spike_train)[$(counter)] == 1")

poisson_const_init = {"counter": 0}