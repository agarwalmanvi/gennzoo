from pygenn.genn_model import create_custom_neuron_class
from pygenn import genn_model
import numpy as np

poisson_const = create_custom_neuron_class(
    "poisson_const",
    param_names=[],
    var_name_types=[("counter", "scalar")],
    sim_code="""
    printf(to_string($(arr)[$(counter)]));
    $(counter) += 1;
    """,
    threshold_condition_code="")

poisson_const_init = {"counter": 0}

model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = 1.0

ssa = model.add_neuron_population("ssa", 1, poisson_const, {}, poisson_const_init)
arr = np.array([1, 2, 3, 4, 5])
ssa.set_extra_global_param("arr", arr)

model.build()
model.load()

while model.timestep < 3.0:
    model.timestep()
