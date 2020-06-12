from pygenn.genn_model import create_custom_neuron_class
from pygenn import genn_model
import numpy as np

poisson_const = create_custom_neuron_class(
    "poisson_const",
    param_names=[],
    var_name_types=[("counter", "int"), ("item", "int")],
    sim_code="""
    $(item) = $(arr)[$(counter)];
    $(counter) += 1;
    """,
    extra_global_params=[("arr", "int*")])

poisson_const_init = {"counter": 0,
                      "item": 0}

model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = 1.0

ssa = model.add_neuron_population("ssa", 1, poisson_const, {}, poisson_const_init)
arr = np.array([1, 2, 3, 4, 5])
ssa.set_extra_global_param("arr", arr)

model.build()
model.load()

while model.timestep < 4.0:

    model.step_time()

    model.pull_var_from_device("ssa", "item")
    i = ssa.vars["item"].view
    print(i)
