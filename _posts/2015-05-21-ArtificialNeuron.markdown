---
layout:     post
title:      "Artificial Neuron"
subtitle:   "Fundamentals of Neural Network I"
date:       2015-05-21 12:00:00
author:     "Laksh Gupta"
header-img: "img/perceptron-bg.jpg"
---

Billions of neuron work together in a highly parallel manner to form the most sophisticated computing device known as the human brain. A single neuron:

- receives the electrical signals from the axons of other neurons through dendrites. Signals can come from different organs such as eyes and ears.
- modulates the signals in various amounts at the synapses between the dendrite and axons.
- fires an output signal only when the total strength of the input signals exceed a certain threshold. This signal is sent further to other neurons.

<p></br></p>
<center>
<img src="{{ site.baseurl }}/img/nn/bioneuron.jpg" alt="Biological Neuron">
<span class="caption text-muted">A Biological Neuron</span>
</center>

More information about a biological neuron can be found on <a href="http://en.wikipedia.org/wiki/Neuron">Wikipedia</a>.

<h2 class="section-heading">Artificial Neuron</h2>
An artificial neuron is a mathematical model of a biological neuron. The steps mentined for a biological neuron can be mapped to an artificial neuron as:

- an artificial neuron receives the input as numerical values rather than the electrical signals. Input can come from different sources such as an image or a text.
- it then multiplies each of the input value by a value called the weight.
- weighted sum is calculated then to represent the total strength of the input signal, and an activation function is applied on the sum to get the output. This output can be sent 
  further to other artificial neurons.



<blockquote>The artificial neuron receives one or more inputs (representing dendrites) and sums them to produce an output (representing a neuron's axon). 
Usually the sums of each node are weighted, and the sum is passed through a non-linear function known as an activation function or transfer function. 
The transfer functions usually have a sigmoid shape, but they may also take the form of other non-linear functions, piecewise linear functions, or step functions. 
They are also often monotonically increasing, continuous, differentiable and bounded.
<p align="right">- <a href="http://en.wikipedia.org/wiki/Artificial_neuron">Wikipedia</a></p>
</blockquote>


