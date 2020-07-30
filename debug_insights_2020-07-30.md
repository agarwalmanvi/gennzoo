Debugging notes for superspike and xor problem.

1. There was a bug where $(lambda) was used instead of $(tau_rms). The exact consequences of this were a bit unclear.
2. The individual learning rates had the strange effect that when one trial was complete with the first spikes in teh middle layer, in subsequent trials without middle layer spikes the residual non-zero value of lambda in the mid layer to out put synapses was enough to lead to similar sized plasticity for whichever output neuron was receiving an error signal in the next few trials. I.e. the first active mid layer neurons would have their synapses strengthened equally to bit output neurons as they received error signals in approximately same frequency.
This led to initial hid2out matrices where connections to both output neurons were very similar in strength for each hid neuron ("horizontal stripes"). This is obviously counter-productive but persisted for many trials.
Core problem: non-zero lambda *and* upsilon mechanism that increased learning updates so that low lambda had same the same effect as high lambda...

Two fixes that both worked:
a) Make a larger tau_rms. I set it to 300, and voila, the amplification of meaningless learning signals after a meaningful learning trial goes away; the connectivity matrices that emerge make more sense. And it seems to perform ok with symmetric feedback as well as random feedback.

b) Additionally or alternatively one can set lambda to 0 between trials. That should work on its own (not tested) bit did certainly work in conjunction with tau_rms==300.

3. (yet unfixed): I noticed that some parts of the code depend on DT= 1ms. That should all be made time step independent.

4. I think there are simpler ways of doing the trial_end_t business. Certainly the modula operation isn't necessary this way and also, as done at the moment this would be called 10 times at a 0.1ms DT.

5. Is there a particular reason why we are doing the massive spike time arrays here instead of simple Poisson groups? I suspect his might be historic, coming from the other experiments where the Poisson trains had to be frozen/ repeatable.

6. I also made some adjustments to the upsilon, epsilon equations to make our code more conformant with the Zenke implementation which I have inspected one more time.
