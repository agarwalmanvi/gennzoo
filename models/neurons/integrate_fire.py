"""
Integrate-and-fire neuron with linear leak
Note that the reset dynamics are specified within `sim_code`,
but typically they should be placed in `reset_code`
Parameters: Vtheta -> Reset threshold
            lambda -> Leak / membrane potential decay constant
            Vrest -> Resting value of membrane potential
            Vreset -> Reset value of membrane potential
Variables: V -> Membrane potential

Neuron model from:
J. M. Brader, W. Senn and S. Fusi,
"Learning Real-World Stimuli in a Neural Network with Spike-Driven Synaptic Dynamics,"
in Neural Computation, vol. 19, no. 11, pp. 2881-2912, 2007,
doi: 10.1162/neco.2007.19.11.2881.
"""

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