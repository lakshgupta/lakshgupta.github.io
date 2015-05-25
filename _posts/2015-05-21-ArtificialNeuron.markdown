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


<center><canvas id="artificialneuron" width="500" heigth="400" markdown="0"></canvas></center>


- $$f(w^Tx) = \phi(\sum\limits_{i=0}^n(w_i x_i))$$ &nbsp;
- $$\phi$$ is our activation function.
- $$x_i$$ are the elements of the input matrix x.
- $$w_i$$ are the elements of the weight matrix y.
- $$y$$ is the output.


An artificial neuron using a step activation function is known as a Perceptron.
<blockquote>In fact, a small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0 to 1. 
That flip may then cause the behaviour of the rest of the network to completely change in some very complicated way. 
So while your "9" might now be classified correctly, the behaviour of the network on all the other images is likely to have completely changed in some hard-to-control way. 
That makes it difficult to see how to gradually modify the weights and biases so that the network gets closer to the desired behaviour.
<p align="right">- <a href="http://neuralnetworksanddeeplearning.com/chap1.html">http://neuralnetworksanddeeplearning.com</a>
</blockquote>


 
<script src="{{ site.baseurl }}/js/nn/canvas.js"></script>
<script src="{{ site.baseurl }}/js/nn/neuron.js"></script>
<script src="{{ site.baseurl }}/js/nn/neuralnet.js"></script>
<script>
//artificial neuron
var _ancanvas = document.getElementById("artificialneuron");
var _anctx = _ancanvas.getContext("2d");
var neuronIn1 = new neuron(_anctx, 50, 40, neuronRadius,"x_0");
var neuronIn2 = new neuron(_anctx, 50, 110, neuronRadius, "x_n");
var	hiddenLayer= new neuron(_anctx, 250, 75, neuronRadius);
_anctx.mathText("f(w^Tx)",250,120,{"text-align": "center"});
var neuronOut = new neuron(_anctx, 350, 75, neuronRadius,"y");
//input to hidden layer
connectLayers([neuronIn1, neuronIn2], [hiddenLayer]);
//hidden to output layer
connectLayers([hiddenLayer], [neuronOut]);

</script>
