# GeNNZoo: Model Zoo for [GeNN](https://github.com/genn-team/genn)

This repository is my project for Google Summer of Code 2020. 

## Setting up your environment
You can find the `environment.yml` to recreate my local development environment in `conda`.
Note that you may have to build `pygenn` from source or from one of the provided wheels, following the instructions [here](https://github.com/genn-team/genn/tree/master/pygenn).

## Navigating the repository
Below, you can find a description of the different folders and files to help you navigate the code. 
1. The __neuron models__ and __weight update models__ are housed in the `models/` directory. 
It contains all the models you need for running the scripts in this repository.
2. A good place to start are the two tutorial notebooks: `Spike-driven Synaptic Plasticity Tutorial` parts I and II. 
They explain the neuron model and weight update model from the paper: 
["Learning Real-World Stimuli in a Neural Network with Spike-Driven Synaptic Dynamics" by J. M. Brader, W. Senn and S. Fusi.](https://ieeexplore.ieee.org/document/6796906).
They also show how to reproduce some of the simulations described in this paper.
3. We also include scripts to reproduce simulations from the paper:
["SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks" by F. Zenke and S. Ganguli](https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086).
 
    a. `timed_spikes_exp1.py` corresponds to the simulation from Fig. 2.<br>
    *Task description:* Teach one output neuron to fire 5 equidistant spikes in a time-window of 500 ms.
    The output neuron receives input from a population of 100 neurons in the form of a frozen Poisson spike train.
    
    b. `timed_spikes_exp2.py` corresponds to the simulation from Fig. 4.<br>
    *Task description:* This is the same task as in `timed_spike_exp1.py` but now with the addition of a hidden layer of 4 neurons to the network topology.
    
    c. `xor.py` corresponds to the simulation from Fig. 5.<br>
    *Task description:* Teach a network to solve the XOR task.
    The input population consists of 100 neurons, divided into three non-overlapping groups: time-reference, input 1, and input 2.
    The hidden layer consists of 100 neurons.
    The output layer consists of 2 output neurons, one for each class.
    
    d. `pattern.py` corresponds to the simulation from Fig. 6.<br>
    *Task description:* Teach a network to produce a predefined spike train.
    The chosen spike train is the pattern given [here](https://github.com/fzenke/pub2018superspike).
    The number of input and output neurons can depend on the chosen spike train.
    The number of hidden neurons can be changed depending on the desired configuration.
    
For each of these scripts, I have also provided inline comments where necessary to make the code clearer.

## Mentors

[Dr. Jamie Knight](https://profiles.sussex.ac.uk/p415734-james-knight), [Prof. Thomas Nowotny](https://profiles.sussex.ac.uk/p206151-thomas-nowotny)
 
    