---
layout:     post
title:      "Perceptron"
subtitle:   "Fundamentals of Neural Network I"
date:       2015-05-21 12:00:00
author:     "Laksh Gupta"
header-img: "img/perceptron-bg.jpg"
---

<p>Billions of neuron work together in a highly parallel manner to form the most sophisticated computing device known as the human brain. A single neuron:
<ul>
  <li>receives the electrical signals from the axons of other neurons through dendrites. Signals can come from different organs such as eyes and ears.</li>
  <li>modulates the signals in various amounts at the synapses between the dendrite and axons.</li>
  <li>fires an output signal only when the total strength of the input signals exceed a certain threshold. This signal is sent further to other neurons.</li>
</ul>

<center>
<img src="{{ site.baseurl }}/img/nn/bioneuron.jpg" alt="Biological Neuron">
<span class="caption text-muted">A Biological Neuron</span>
</center>

More information about a biological neuron can be found on <a href="http://en.wikipedia.org/wiki/Neuron">Wikipedia</a>.
</p>

<h2 class="section-heading">The Perceptron</h2>
<p>
The perceptron is a mathematical model of a biological neuron. The steps mentined for a biological neuron can be mapped to a perceptron as:
<ul>
  <li>a perceptron receives the input as numerical values rather than the electrical signals. Input can come from different sources such as an image or a text.</li>
  <li>it then multiplies each of the input value by a value called the weight.</li>
  <li>weighted sum is calculated then to represent the total strength of the input signal, and an activation function is applied on the sum to get the output. This output can be sent 
  further to other perceptrons.</li>
</ul>

<blockquote>The artificial neuron receives one or more inputs (representing dendrites) and sums them to produce an output (representing a neuron's axon). 
Usually the sums of each node are weighted, and the sum is passed through a non-linear function known as an activation function or transfer function. 
The transfer functions usually have a sigmoid shape, but they may also take the form of other non-linear functions, piecewise linear functions, or step functions. 
They are also often monotonically increasing, continuous, differentiable and bounded.
<p align="right">- <a href="http://en.wikipedia.org/wiki/Artificial_neuron">Wikipedia</a></p>
</blockquote>

Here is an example MathJax inline rendering \\( 1/x^{2} \\), and here is a block rendering:

\\[ \frac{1}{n^{2}} \\]


$$\frac{1}{n^{2}}$$


$$
J(\theta) = - \frac{1}{m}[\sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)} \log (h_\theta(x^{(i)}))_k + (1-y_k^{(i)}) \log(1-(h_\theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{S_l-1}\sum_{j=1}^{S_l-1}(\theta_{ji}^{(l)})^2
$$


$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$


</p>



