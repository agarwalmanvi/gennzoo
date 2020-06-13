'''
Sample script for setting iterable as extra_global_param and accessing things inside iterable
'''

from pygenn.genn_model import create_custom_neuron_class
from pygenn import genn_model
import numpy as np

poisson_const = create_custom_neuron_class(
    "poisson_const",
    param_names=[],
    var_name_types=[("counter", "int"), ("item", "int"), ("flag", "int")],
    sim_code="""
    $(item) = $(arr)[$(counter)];
    if ( $(item) == 1 ) {
        $(flag) = 1;
    } else {
        $(flag) = 0;
    }
    $(counter) += 1;
    """,
    extra_global_params=[("arr", "int*")])

poisson_const_init = {"counter": 0,
                      "item": 0,
                      "flag": 2}

model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = 1.0

ssa = model.add_neuron_population("ssa", 1, poisson_const, {}, poisson_const_init)
arr = np.array([0, 0, 1, 0, 1])
ssa.set_extra_global_param("arr", arr)

model.build()
model.load()

while model.timestep < 5.0:

    print("Timestep " + str(model.timestep))

    model.step_time()

    model.pull_var_from_device("ssa", "flag")
    i = ssa.vars["flag"].view
    print(i)
