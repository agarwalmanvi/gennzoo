import numpy as np
from pygenn.genn_model import GeNNModel, create_custom_current_source_class
from models.neurons.lif_superspike import lif_model, LIF_PARAMS, lif_init
import matplotlib.pyplot as plt

# TODO Non-numeric arguments not supported error for spike_train init

# interval = 500
# freq = 50
# spike_dt = 0.001
# N_INPUT = 1
#
# spike_train = np.random.random_sample((N_INPUT, interval))
# spike_train = (spike_train < freq * spike_dt).astype(int)

cs_model = create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

TIMESTEP = 1.0

model = GeNNModel("float", "superspike")
model.dT = TIMESTEP

n = model.add_neuron_population("input", 1, lif_model, LIF_PARAMS, lif_init)

current_input = model.add_current_source("current_input", cs_model,
                                         "input", {}, {"magnitude": 2.0})

model.build()
model.load()

PRESENT_TIMESTEPS = 200

voltage = np.array([])
z_tilda = np.array([])
sigma_prime = np.array([])

while model.timestep < PRESENT_TIMESTEPS:

    model.step_time()

    model.pull_var_from_device("input", "V")
    voltage = np.append(voltage, n.vars["V"].view)

    model.pull_var_from_device("input", "z_tilda")
    z_tilda = np.append(z_tilda, n.vars["z_tilda"].view)

    model.pull_var_from_device("input", "sigma_prime")
    sigma_prime = np.append(sigma_prime, n.vars["sigma_prime"].view)

timesteps = list(range(PRESENT_TIMESTEPS))
fig, axes = plt.subplots(3, sharex=True)
axes[0].plot(timesteps, voltage)
axes[0].set_title("Voltage")
axes[1].plot(timesteps, z_tilda)
axes[1].set_title("Presynaptic Trace")
axes[2].plot(timesteps, sigma_prime)
axes[2].set_title("Partial derivative")
fig.tight_layout(pad=1.0)
plt.savefig("superspike.png")
