---
title: "Neural Network"
subtitle: "activation function, cost function, regularization, backpropagation, gradient descent"
description: "Building neural networks for MNIST digit recognition; activation, cost, and backpropagation."
date: 2015-06-12T12:00:00
author: "Laksh Gupta"
tags: ["neural-network", "machine-learning"]
---

<div tabindex="-1" id="notebook" class="border-box-sizing">
  <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">The Problem</h2><p>Single neuron has limited computational power and hence we need a way to build a network of neurons to make a more complex model. In this post we will look into how to construct a neural network and try to solve the handwritten digit recognition problem. The goal is to decide which digit it represents when given a new image.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Understanding the Data</h2><p>We'll use the <a href="http://yann.lecun.com/exdb/mnist/">MNIST dataset</a>. Luckily, <a href="https://github.com/johnmyleswhite/MNIST.jl">John Myles White</a> has already created a package to import this dataset in Julia. The MNIST dataset provides a training set of 60,000 handwritten digits and a test set of 10,000 handwritten digits. Each of the image has a size of 28×28 pixels. <img src="/notebooks/img/nn/MNIST_digits.png" alt="MNIST"></p>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ======================</span>
<span class="c"># network setup</span>
<span class="c"># ======================</span>
<span class="n">inputLayerSize</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="mi">1</span><span class="p">);</span> <span class="c"># number of input features: 784</span>
<span class="n">hiddenLayerSize</span> <span class="o">=</span> <span class="mi">25</span><span class="p">;</span> <span class="c"># variable</span>
<span class="n">outputLayerSize</span> <span class="o">=</span> <span class="mi">10</span><span class="p">;</span> <span class="c"># number of output classes</span>
<span class="c"># since we are doing multiclass classification: more than one output neurons</span>
<span class="c"># representing each output as an array of size of the output layer</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">outputLayerSize</span><span class="p">,</span><span class="n">m</span><span class="p">);</span> <span class="c">#Y:(10,60000)</span>
<span class="k">for</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">m</span>
    <span class="n">Y</span><span class="p">[</span><span class="kt">Int64</span><span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">+</span><span class="mi">1</span><span class="p">),</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The appraoch to train a neural network is similar to what we have discussed in the <a href="/2015/05/21/ArtificialNeuron/">neuron post</a>.  But since now we have a network of neurons, the way we follow the steps is a bit different. We'll use the <a href="http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf">backpropagation algorithm</a>.</p>
<blockquote><p>Backpropagation works by approximating the non-linear relationship between the input and the output by adjusting the weight values internally. The operations of the Backpropagation neural networks can be divided into two steps: feedforward and Backpropagation. In the feedforward step, an input pattern is applied to the input layer and its effect propagates, layer by layer, through the network until an output is produced. The network's actual output value is then compared to the expected output, and an error signal is computed for each of the output nodes. Since all the hidden nodes have, to some degree, contributed to the errors evident in the output layer, the output error signals are transmitted backwards from the output layer to each node in the hidden layer that immediately contributed to the output layer. This process is then repeated, layer by layer, until each node in the network has received an error signal that describes its relative contribution to the overall error. Once the error signal for each node has been determined, the errors are then used by the nodes to update the values for each connection weights until the network converges to a state that allows all the training patterns to be encoded.</p>
<p>- <a href="http://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html">www.cse.unsw.edu.au</a></p>
</blockquote>
<p>We'll discuss more about the backpropagation algorithm later but first let's collect the simple tools which are required for training a neural network.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 class="section-heading">Activation Function: $g$</h4><p>The activation function of artificial neurons have to be differentiable and their derivative has to be non-zero so that the gradient descent learning algorithm can be applied. Considering <a href="/2015/05/27/LinearRegression/">linear regression</a>, using a linear activation function does not give us much advantage here. Linear function applied to a linear function is itself a linear function, and hence both the functions can be replaced by a single linear function. Moreover real world problems are generally more complex. A linear activation function may not be a good fit for the dataset we have. Therefore if the data we wish to model is non-linear then we need to account for that in our model. Sigmoid activation function is one of the reasonably good non-linear activation functions which we could use in our neural network.</p>
$$sigmoid(z) = 1/(1 + e^{-z})$$<p><img src="/notebooks/img/nn/sigmoidGraph.png" alt="sigmoid"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ==============================================</span>
<span class="c"># activation function: computes the sigmoid of z </span>
<span class="c"># z is the weighted sum of inputs</span>
<span class="c"># ==============================================</span>
<span class="k">function</span><span class="nf"> sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">)</span>
    <span class="n">g</span> <span class="o">=</span> <span class="mf">1.0</span> <span class="o">./</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">+</span> <span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">z</span><span class="p">));</span>
    <span class="k">return</span> <span class="n">g</span><span class="p">;</span>
<span class="k">end</span>

<span class="c"># computes the gradient of the sigmoid function evaluated at z</span>
<span class="k">function</span><span class="nf"> sigmoidGradient</span><span class="p">(</span><span class="n">z</span><span class="p">)</span>
  <span class="k">return</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">)</span><span class="o">.*</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">));</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>
