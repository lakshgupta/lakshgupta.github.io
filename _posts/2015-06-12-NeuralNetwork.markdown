---
layout:     post
title:      "Neural Network"
subtitle:   "handwritten digit recognition"
date:       2015-06-12 12:00:00
author:     "Laksh Gupta"
header-img: "img/sd3-bg.jpg"
---

  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">The Problem</h2>

<p>Single neuron has limited computational power and hence we need a way to build a network of neurons to make a more complex model. In this post we will look into how to construct a neural network and try to solve the handwritten digit recognition problem. The goal is to decide which digit it represents when given a new image.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Understanding the Data</h2>

<p>We'll use the <a href="http://yann.lecun.com/exdb/mnist/">MNIST dataset</a>. Luckily, <a href="https://github.com/johnmyleswhite/MNIST.jl">John Myles White</a> has already created a package to import this dataset in Julia. The MNIST dataset provides a training set of 60,000 handwritten digits and a test set of 10,000 handwritten digits. Each of the image has a size of 28×28 pixels. <img src="{{ site.baseurl }}/img/nn/MNIST_digits.png" alt="MNIST"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">#Pkg.update();</span>
<span class="n">Pkg</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="s">&quot;MNIST&quot;</span><span class="p">);</span>
<span class="n">Pkg</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="s">&quot;PyPlot&quot;</span><span class="p">)</span>
<span class="c">#Pkg.installed();</span>
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
<p>For plotting, PyPlot is a good option. It provides a Julia interface to the Matplotlib plotting library from Python.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">using</span> <span class="n">MNIST</span>
<span class="k">using</span> <span class="n">PyPlot</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>Qt: Untested Windows version 10.0 detected!
INFO: Loading help data...
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ===================</span>
<span class="c"># load training data</span>
<span class="c"># ===================</span>
<span class="n">X</span><span class="p">,</span><span class="n">y</span> <span class="o">=</span> <span class="n">traindata</span><span class="p">();</span> <span class="c">#X:(784x60000), y:(60000x1)</span>
<span class="n">m</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="mi">2</span><span class="p">);</span> <span class="c"># number of inputs</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[3]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>60000</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Training a model</h2>

<p>We want to train a neural network with one input layer, one hidden layer and one output layer to recognize handwritten digits. Since the dataset contains 28×28 pixel images, our neural network will have $28*28=784$ input neurons, a variable number of hidden neurons and $10$ output neurons.</p>
<p><img src="{{ site.baseurl }}/img/nn/nn_basic.png" alt="2-layer-neuralNetwork"></p>

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
<span class="n">inputLayerSize</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="mi">1</span><span class="p">);</span> <span class="c"># number of input features: 784</span>
<span class="n">hiddenLayerSize</span> <span class="o">=</span> <span class="mi">25</span><span class="p">;</span> <span class="c"># variable</span>
<span class="n">outputLayerSize</span> <span class="o">=</span> <span class="mi">10</span><span class="p">;</span> <span class="c"># number of output classes</span>
<span class="c"># since we are doing multiclass classification: more than one output neurons</span>
<span class="c"># representing each output as an array of size of the output layer</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">outputLayerSize</span><span class="p">,</span> <span class="n">m</span><span class="p">);</span> <span class="c">#Y:(10,60000)</span>
<span class="k">for</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">m</span>
    <span class="n">Y</span><span class="p">[</span><span class="n">y</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">+</span><span class="mi">1</span><span class="p">,</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span>
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
<p>The appraoch to train a neural network is similar to what we have discussed in the <a href="http://lakshgupta.github.io/2015/05/21/ArtificialNeuron/">neuron post</a>.  But since now we have a network of neurons, the way we follow the steps is a bit different. We'll use the <a href="http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf">backpropagation algorithm</a>.</p>
<blockquote><p>Backpropagation works by approximating the non-linear relationship between the input and the output by adjusting the weight values internally. 
The operations of the Backpropagation neural networks can be divided into two steps: feedforward and Backpropagation. In the feedforward step, an input pattern is applied to the input layer and its effect propagates, layer by layer, through the network until an output is produced. The network's actual output value is then compared to the expected output, and an error signal is computed for each of the output nodes. Since all the hidden nodes have, to some degree, contributed to the errors evident in the output layer, the output error signals are transmitted backwards from the output layer to each node in the hidden layer that immediately contributed to the output layer. This process is then repeated, layer by layer, until each node in the network has received an error signal that describes its relative contribution to the overall error.
Once the error signal for each node has been determined, the errors are then used by the nodes to update the values for each connection weights until the network converges to a state that allows all the training patterns to be encoded.</p>
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
<h4 class="section-heading">Activation Function: $g$</h4>

<p>The activation function of artificial neurons have to be differentiable and their derivative has to be non-zero so that the gradient descent learning algorithm can be applied. Considering <a href="http://lakshgupta.github.io/2015/05/27/LinearRegression/">linear regression</a>, using a linear activation function does not give us much advantage here. Linear function applied to a linear function is itself a linear function, and hence both the functions can be replaced by a single linear function. Moreover real world problems are generally more complex. A linear activation function may not be a good fit for the dataset we have. Therefore if the data we wish to model is non-linear then we need to account for that in our model. Sigmoid activation function is one of the reasonably good non-linear activation functions which we could use in our neural network.</p>
$$sigmoid(z) = 1/(1 + e^{-z})$$<p><img src="{{ site.baseurl }}/img/nn/sigmoidGraph.png" alt="sigmoid"></p>

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

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[5]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>sigmoidGradient (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 class="section-heading">Cost Function: $J$</h4>

<p>We used squared error (SE) cost function for performing <a href="http://lakshgupta.github.io/2015/05/27/LinearRegression/">linear regression</a>. But for training the neural network we'll use cross entropy (CE) cost function instead.</p>
<blockquote><p>The experimental results have shown that, in a comparable environment and with randomly initialized weights, the CE criterion allows to find a better local optimum than the SE criterion. The training of the SE system quickly got stuck in a worse local optimum where the gradient vanished and no further reduction of the classification errors was possible.</p>
<p>- <a href="https://www-i6.informatik.rwth-aachen.de/publications/download/861/GolikPavelDoetschPatrickNeyHermann--Cross-Entropyvs.SquaredErrorTrainingaTheoreticalExperimentalComparison--2013.pdf">P. Golik, P. Doetsch, and H. Ney</a></p>
</blockquote>
<p>So considering:</p>
$$J(\theta) = \frac{1}{m}(\sum_{i=1}^{m}cost(h_{\theta}(x^{(i)}),y^{(i)}))$$<p>where:</p>
<ul>
<li>$h_{\theta}(x^{(i)})$ is the predicted value (hypothesis)</li>
<li>$y^{(i)}$ is the actual value (truth), and
$$\begin{eqnarray}
cost(h_{\theta}(x^{(i)}),y^{(i)})&=&\left\{
\begin{array}{l l}      
  -\log(h_{\theta}(x^{(i)}))   &   \mathrm{if} \: y=1 \\
  -\log(1-h_{\theta}(x^{(i)})) &  \mathrm{if}  \: y=0
\end{array}\right.,  \: h_{\theta}(x^{(i)})\in(0,1) \\ \nonumber
&=& - y^{(i)}\log{h_{\theta}(x^{(i)})} - (1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))
\end{eqnarray}$$</li>
</ul>
<p>Hence our cost function becomes:
$$J(\theta) = -\frac{1}{m}[\sum_{i=1}^{m} ( y^{(i)}\log{h_{\theta}(x^{(i)})} + (1-y^{(i)})\log({1-h_{\theta}(x^{(i)}))})]$$</p>
<p>We don't sum over the bias terms hence starting at 1 for the summation. The above equation for the cost function will work if we have a single neuron in the output layer. Let's generalize this cost function so that we could use it for $k$ neurons in the output layer. 
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} \sum_{i=1}^{k}[ y^{(i)}_k\log{(h_{\theta}(x^{(i)})_k)} + (1-y^{(i)}_k)\log({1-(h_{\theta}(x^{(i)}))_k)}]$$</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 class="section-heading">Regularization: $L^2$</h4>

<p>Regularization helps us in handling the problem of overfitting. Most regularization approaches add a parameter norm penalty $\Omega(\theta)$ to the loss function $J$ to achieve better generalization of the model. In case of $L^2$ regularization, also known as weight decay, the penalty is equal to the sum of the square of all of the weight vectors.</p>
$$\Omega(\theta) = \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}((\theta_{ji}^l)^2)$$<p></p>
<p>where</p>
<ul>
<li>$\lambda>0$, is known as the regularization parameter</li>
<li>$m$ is the size of our training set</li>
<li>$L$ in the equation is the layer number</li>
<li>$s$ is the neuron unit in the corresponding layer</li>
</ul>
<blockquote><p>Regularizers work by trading increased bias for reduced variance. An effective regularizer is one that makes a proﬁtable trade, that is it reduces variance signiﬁcantly while not overly increasing the bias.</p>
<p>- <a href="http://www.iro.umontreal.ca/~bengioy/dlbook/regularization.html">Yoshua Bengio, Ian Goodfellow and Aaron Courville</a></p>
</blockquote>
<p>The Wikipedia has a descent article on the <a href="https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff">bias-variance tradeoff</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># weight regularization parameter</span>
<span class="n">lambda</span> <span class="o">=</span> <span class="mi">3</span><span class="p">;</span> 

<span class="c"># ===============================================</span>
<span class="c"># cross entropy cost function with regularizarion</span>
<span class="c"># ===============================================</span>
<span class="k">function</span><span class="nf"> costFunction</span><span class="p">(</span><span class="n">truth</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
    <span class="n">cost</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">m</span><span class="p">,</span><span class="mi">1</span><span class="p">);</span>
    <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">m</span>
        <span class="n">cost</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">=</span> <span class="p">(</span><span class="o">-</span><span class="n">Y</span><span class="p">[:,</span><span class="n">i</span><span class="p">]</span><span class="o">&#39;*</span><span class="n">log</span><span class="p">(</span><span class="n">prediction</span><span class="p">[:,</span><span class="n">i</span><span class="p">]))</span> <span class="o">-</span> <span class="p">((</span><span class="mi">1</span><span class="o">-</span><span class="n">Y</span><span class="p">[:,</span><span class="n">i</span><span class="p">]</span><span class="o">&#39;</span><span class="p">)</span><span class="o">*</span><span class="n">log</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">prediction</span><span class="p">[:,</span><span class="n">i</span><span class="p">]));</span>
    <span class="k">end</span>
    <span class="c"># regularization term</span>
    <span class="n">regularization</span> <span class="o">=</span> <span class="p">(</span><span class="n">lambda</span><span class="o">/</span><span class="p">(</span><span class="mi">2</span><span class="o">*</span><span class="n">m</span><span class="p">))</span><span class="o">*</span><span class="p">(</span><span class="n">sum</span><span class="p">(</span><span class="n">sum</span><span class="p">(</span><span class="n">Theta1</span><span class="p">[</span><span class="mi">2</span><span class="p">:</span><span class="k">end</span><span class="p">,:]</span><span class="o">.^</span><span class="mi">2</span><span class="p">))</span> <span class="o">+</span> <span class="n">sum</span><span class="p">(</span><span class="n">sum</span><span class="p">(</span><span class="n">Theta2</span><span class="p">[</span><span class="mi">2</span><span class="p">:</span><span class="k">end</span><span class="p">,:]</span><span class="o">.^</span><span class="mi">2</span><span class="p">)));</span>

    <span class="k">return</span> <span class="p">(</span><span class="mi">1</span><span class="o">/</span><span class="n">m</span><span class="p">)</span><span class="o">*</span><span class="n">sum</span><span class="p">(</span><span class="n">cost</span><span class="p">)</span> <span class="o">+</span> <span class="n">regularization</span><span class="p">;</span> <span class="c"># regularized cost</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[6]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>costFunction (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 class="section-heading">Backpropagation</h4>

<p>Despite the name, <a href="http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm">backpropagation algorithm</a> consist of two phases:</p>
<ul>
<li>Feedforward </li>
<li>Backpropagation</li>
</ul>
<p>The feedforward process is the same process we have been following in the previous posts. Using the feedforward process we calculate the weighted sum of inputs and apply the activation function to get an output as we move from layer to layer. In the end we come up with a output activation which could have some error as compared to the actual values. To have the output as close as possible to the actual values we use the backpropagation process to tune the weights.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="{{ site.baseurl }}/img/nn/ff_mnist.png" alt="2-layer-neuralNetwork-feedforward"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Backpropagation is a way of computing gradients of expressions through recursive application of chain rule. We start from the output layer and go backwards calulating the gradient on the activations for each layer till the first hidden layer.</p>
<blockquote><p>From these gradients, which can be interpreted as an indication of how each layer’s output should change to reduce error, one can obtain the gradient on the parameters of each layer. The gradients on weights and biases can be immediately used as part of a stochastic gradient update (performing the update right after the gradients havebeen computed) or used with other gradient-based optimization methods.</p>
<p align="right">- [Yoshua Bengio, Ian Goodfellow and Aaron Courville](http://www.iro.umontreal.ca/~bengioy/dlbook/regularization.html)</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="{{ site.baseurl }}/img/nn/bp_mnist.png" alt="2-layer-neuralNetwork-backpropagation"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Since we are learning/tuning weights ($\theta$), we want to evaluate: $\dfrac{\partial J}{\partial \theta^{(l)}}$, with $l$ as the layer number. Using the chain rule, we can solve the above partial derivative as:
$$\dfrac{\partial J}{\partial \theta^{(l)}} = \underbrace{\dfrac{\partial J}{\partial z^{(l+1)}}}_{1} \underbrace{\dfrac{\partial z^{(l+1)}}{\partial \theta^{(l)}}}_{2}$$</p>
<p>here, $z$ represents the input signal to a neuron which is the weighted sum of the outputs from the previous layer's neurons. Hence $(2)$ becomes:</p>
$$\dfrac{\partial z^{(l+1)}}{\partial \theta^{(l)}} = a^{(l)}$$<p>where, $a$ is the value from applying activation function $g$ to $z$. Now let's look at $(1)$ and represent it as $\delta$. Since we start backpropagation from the last output layer, we can calculate the change in cost w.r.t weights as:
$$ \dfrac{\partial J}{\partial \theta^{(2)}} = \underbrace{\dfrac{\partial J}{\partial z^{(3)}}}_{\delta^{(3)}} \underbrace{\dfrac{\partial z^{(3)}}{\partial \theta^{(2)}}}_{a^{(2)}}$$</p>
<p>where,
$$ \begin{eqnarray}
\delta^{(3)} &=& \dfrac{\partial J}{\partial z^{(3)}} \\
&=& -[\frac{yg'(z^{(3)})}{g(z^{(3)})} + \frac{(1-y)(-g'(z^{(3)}))}{1-g(z^{(3)})}] \\
&=& g(z^{(3)}) - y \\ \\
&&(\text{for sigmoid, $g'(z) = g(z)(1-g(z))$})
\end{eqnarray}$$</p>
<p>In the squared error cost function, $\delta^{(3)}$ would have a factor of $g'(z^{(3)})$. This means that for a large difference between the truth and the hypothesis, the sigmoid gradient would become very low (sigmoid curve is flat at the top) and hence the learning of our model would be slow. Using the cross entropy cost function also saves us from that problem. In the current case the larger the error, the faster the neuron will learn.</p>
<p>Similarly for the hidden layer we have:
$$ \dfrac{\partial J}{\partial \theta^{(1)}} = \underbrace{\dfrac{\partial J}{\partial z^{(2)}}}_{\delta^{(2)}} \underbrace{\dfrac{\partial z^{(2)}}{\partial \theta^{(1)}}}_{a^{(1)}}$$</p>
<p>where,
$$ \begin{eqnarray}
\delta^{(2)} &=& \dfrac{\partial J}{\partial z^{(2)}} \\
&=& \dfrac{\partial J}{\partial z^{(3)}} \dfrac{\partial z^{(3)}}{\partial g(z^{(2)})} \dfrac{\partial g(z^{(2)})}{\partial z^{(2)}} \\
&=& \delta^{(3)} \theta^{(2)} g'(z^{(2)})
\end{eqnarray}$$</p>
<p>The equations above may require special handling in order to perform matrix operations but basically we saw how the chain rule can be applied for the backpropagation algorithm.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We now have all the components for $\dfrac{\partial J}{\partial \theta^{l}}$, hence we can update the weights as:</p>
$$\theta^{(l)} \leftarrow \theta^{(l)} - \frac{\alpha}{m} \dfrac{\partial J}{\partial \theta^{l}}$$<p>If the original cost function included a regularization term then we need to take it into account as well while taking the derivatives. Hence $\dfrac{\partial J}{\partial \theta^{l}}$ would also include the derivative of the regularization term, i.e. $\frac{\lambda}{m}\theta^{(l)}$.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># including one bias neuron in input layer</span>
<span class="c"># weights for the links connecting input layer to the hidden layer</span>
<span class="n">Theta1</span> <span class="o">=</span> <span class="n">randn</span><span class="p">(</span><span class="n">inputLayerSize</span><span class="o">+</span><span class="mi">1</span><span class="p">,</span> <span class="n">hiddenLayerSize</span><span class="p">);</span> <span class="c">#(785x25)</span>
<span class="c"># including one bias neuron in hidden layer</span>
<span class="c"># weights for the links connecting hidden layer to the output layer</span>
<span class="n">Theta2</span> <span class="o">=</span> <span class="n">randn</span><span class="p">(</span><span class="n">hiddenLayerSize</span><span class="o">+</span><span class="mi">1</span><span class="p">,</span> <span class="n">outputLayerSize</span><span class="p">);</span> <span class="c">#(26x10)</span>
<span class="c"># learning rate</span>
<span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.9</span><span class="p">;</span>
<span class="c"># number of iterations</span>
<span class="n">epoch</span> <span class="o">=</span> <span class="mi">1500</span><span class="p">;</span>
<span class="c"># cost per epoch</span>
<span class="n">J</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">epoch</span><span class="p">,</span><span class="mi">1</span><span class="p">);</span>
<span class="c"># ====================================================================</span>
<span class="c"># Train the neural network using feedforward-backpropagation algorithm</span>
<span class="c"># ====================================================================</span>
<span class="k">for</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">epoch</span>
    <span class="c"># Feedforward #</span>
    <span class="n">a1</span> <span class="o">=</span> <span class="p">[</span><span class="n">ones</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">m</span><span class="p">),</span> <span class="n">X</span><span class="p">];</span> <span class="c"># add one bias element (785x60000)</span>
    <span class="n">z2</span> <span class="o">=</span> <span class="n">Theta1</span><span class="o">&#39;*</span><span class="n">a1</span><span class="p">;</span> <span class="c">#(25x60000)</span>
    <span class="n">a2</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">z2</span><span class="p">);</span>
    <span class="n">a2</span> <span class="o">=</span> <span class="p">[</span><span class="n">ones</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">size</span><span class="p">(</span><span class="n">a2</span><span class="p">,</span><span class="mi">2</span><span class="p">)),</span> <span class="n">a2</span><span class="p">];</span> <span class="c"># add one bias element (26x60000)</span>
    <span class="n">z3</span> <span class="o">=</span> <span class="n">Theta2</span><span class="o">&#39;*</span><span class="n">a2</span><span class="p">;</span> <span class="c">#(10x60000)</span>
    <span class="n">a3</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">z3</span><span class="p">);</span>
    
    <span class="c"># cost </span>
    <span class="n">J</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">=</span> <span class="n">costFunction</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span> <span class="n">a3</span><span class="p">);</span>
    
    <span class="c"># Backpropagation process #</span>
    <span class="n">delta3</span> <span class="o">=</span> <span class="p">(</span><span class="n">a3</span> <span class="o">-</span> <span class="n">Y</span><span class="p">);</span> <span class="c">#(10x60000)</span>
    <span class="n">delta2</span> <span class="o">=</span> <span class="p">(</span><span class="n">Theta2</span><span class="p">[</span><span class="mi">2</span><span class="p">:</span><span class="k">end</span><span class="p">,:]</span><span class="o">*</span><span class="n">delta3</span><span class="p">)</span><span class="o">.*</span><span class="n">sigmoidGradient</span><span class="p">(</span><span class="n">z2</span><span class="p">)</span> <span class="p">;</span> <span class="c">#(25x10)*(10x60000).*(25x60000)</span>
        
    <span class="c">#update weights</span>
    <span class="n">reg_theta2</span> <span class="o">=</span> <span class="p">(</span><span class="n">lambda</span><span class="o">*</span><span class="n">Theta2</span><span class="p">);</span>
    <span class="n">reg_theta2</span><span class="p">[</span><span class="mi">1</span><span class="p">,:]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="n">Theta2</span> <span class="o">=</span> <span class="n">Theta2</span> <span class="o">-</span> <span class="n">alpha</span><span class="o">*</span><span class="p">(</span><span class="n">a2</span><span class="o">*</span><span class="n">delta3</span><span class="o">&#39;</span> <span class="o">+</span> <span class="n">reg_theta2</span><span class="p">)</span><span class="o">/</span><span class="n">m</span><span class="p">;</span> 
    
    <span class="n">reg_theta1</span> <span class="o">=</span> <span class="p">(</span><span class="n">lambda</span><span class="o">*</span><span class="n">Theta1</span><span class="p">);</span>
    <span class="n">reg_theta1</span><span class="p">[</span><span class="mi">1</span><span class="p">,:]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="n">Theta1</span> <span class="o">=</span> <span class="n">Theta1</span> <span class="o">-</span> <span class="n">alpha</span><span class="o">*</span><span class="p">(</span><span class="n">a1</span><span class="o">*</span><span class="n">delta2</span><span class="o">&#39;</span> <span class="o">+</span> <span class="n">reg_theta1</span><span class="p">)</span><span class="o">/</span><span class="n">m</span><span class="p">;</span>
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
<p>If our implementation is correct, the cost of the predicted output after each iteration should drop. I'll cover another method (gradient check) in another post to validate our implementation. But for now let's check by plotting the cost per iteration.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># plot the cost per iteration</span>
<span class="n">plot</span><span class="p">(</span><span class="mi">1</span><span class="p">:</span><span class="n">length</span><span class="p">(</span><span class="n">J</span><span class="p">),</span> <span class="n">J</span><span class="p">)</span>
<span class="n">xlabel</span><span class="p">(</span><span class="s">&quot;Iterations&quot;</span><span class="p">)</span>
<span class="n">ylabel</span><span class="p">(</span><span class="s">&quot;Cost&quot;</span><span class="p">)</span>
<span class="n">grid</span><span class="p">(</span><span class="s">&quot;on&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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dtu3b58WLlyoyZMnq2fPnkpJSdHDDz/c5GNbos8XX3yhSy+9VL1791ZGRobOOOMMbdq0qdmeX0s5mm6e52nZsmWKRqMaOHCgMjIyNHLkSN1yyy1NXhNAatvdTP6uxRw4cEDDhw9XSkqK7rrrriYf05abSWbdamtrdd9992n06NFKT09Xr1699L3vfU9vv/12o8e2ajevjTvvvPO81NRULycnx3vwwQe9iRMneqmpqd7atWuDHq1VTZ8+3evfv7935ZVXenl5ed7NN9/sHXPMMV5GRob37rvv1j3u008/9Xr16uV9/etf9+69917v1ltv9Xr27OmNHj3aq66ujvua999/vxeJRLyZM2d6f/jDH7w5c+Z4kUjEy83Nbe2n1yo+/fRTLz093cvIyPBGjhzZ6GN0q/f88897HTp08CZMmODdc8893h/+8Afvl7/8pbdgwYK6x9As3ttvv+2lpaV5Q4YM8XJzc70HH3zQy8rK8iKRiDdt2rS6x4W12z//+U8vEol4gwcP9k4//XQvEol4Dz/8cKPHtUSfmpoab+LEiV5GRoZ30003eb/97W+9ESNGeF27dvW2bNnSos/7qzqabuXl5V4kEvEmTpzo3Xrrrd4f/vAHb+7cuV67du28008/vdHXbOvdjvbv2qHuuusuLyMjw4tEIt5dd93V6ONtvZnnmXW76KKLvNTUVO+nP/2pl5eX5/3617/2srKyvNWrV8c9rrW7temleMOGDY3+glZVVXnHHXecN3HixAAna31vvPGGd+DAgbj7tmzZ4nXs2NGbPXt23X0///nPvc6dO3uffvpp3X2rV6/2IpGI98ADD9TdV1FR4WVmZnpTp06N+5qzZ8/2MjIyvD179rTQMwnOrFmzvO9///veaaed5n3jG9+I+xjd6n355Zde3759venTpx/xcTSLd/3113uRSMR7//334+6/6KKLvEgk4n3xxRee54W32/79+72dO3d6nud5b7311mH/D7cl+qxcudKLRCLeE088UXffrl27vB49engXXHBBsz3HlnA03aqrq71169Y1+tybbrrJi0QicYtKGLod7d+1mJ07d3rdu3f3br755iaX4jA087yj7xZ7jgUFBUf8ekF0a9NL8fz5873U1FSvvLw87v7bbrvNi0Qi3rZt2wKaLHmMHTvWO/nkk+t+3adPH2/WrFmNHjds2DDv+9//ft2v/+d//seLRCLes88+G/e4devWeZFIxHv00UdbbugAvPrqq1779u29d9991/vud7/b6JViutW77777vEgk4hUXF3ue53l79+71ampqGj2OZvEWL17sRSIRr7S0NO7+BQsWeO3bt/cqKio8z6Ob53nem2++edj/w22JPjNnzvT69evX6GtedtllXufOnRu9Ap2sjtStKW+//bYXiUS8pUuX1t0Xtm5H0ywrK8sbP3583SulDZfisDXzvCN3O/XUU73x48d7nue/wrt3794mv0YQ3dr0McWbNm3S8ccfr4yMjLj7x40bJ0n629/+FsRYScPzPO3cuVO9evWSJH322WfatWuXTj755EaPHTduXNyxObGfN3zs2LFjlZKS0qba1tTUaN68ebrkkks0YsSIRh+nW7zVq1era9eu+vTTTzVs2DB16dJF3bp10y9+8Yu64xNp1tjcuXPVt29f/eQnP9HmzZv16aefauXKlbr//vt1xRVXqFOnTnRLoKX6bNq0SWPHjm3ya1ZUVOgf//hHcz2FpLJjxw5Jqvv/CIluDW3cuFGPPPKI7rnnnsM+hmb1ysrK9Oabb+rkk0/W9ddfr27duqlLly4aOnSoVq1aFffYILq16aW4pKRE/fr1a3R/7L7t27e39khJZfny5dq+fbtmzZolye8l6bDNdu/erQMHDtQ9tl27dnH/YylJHTp0UGZmZptqe//992vr1q1avHhxkx+nW7wtW7bo4MGD+sEPfqApU6boySef1Ny5c3X//fcrKytLEs2a0r9/f73++usqLi7WmDFjNGjQIJ1//vm64oor6t64Q7cja6k+Yf3/kiVLlqhbt26aMmVK3X10q+d5nubNm6fzzjtPp5566mEfR7N6//u//yvP87RixQotW7ZMd955p5YvX67evXvrvPPO0/PPP1/32CC6OXlFu6NVWVmptLS0Rvd37Nix7uNhVVxcrMsvv1wTJ07URRddJKm+R6JmqampqqysVIcOHZr82mlpaW2m7eeff67/+q//0n/9138pMzOzycfQLd7evXtVUVGhn//853WvnvzgBz9QdXW1fv/73+umm26iWRN27txZt3w8+OCDyszM1DPPPKNbbrlFffv21eWXX063BFqqT1VVVej+v+TWW2/VSy+9pPvuu09du3atu59u9ZYtW6Z3331XTz755BEfR7N6e/fulSTt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"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Prediction and Accuracy</h2>

<p>After training the model we'll check how well our model has learned all the weights to make a prediction. So we'll evaluate the performance on the train dataset first. We do it by following a way similar to the feedforward process. Consider that we now do not have randomly initialized weights but rather obtained after going through the backpropagation algorithm.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">##########################################################################</span>
<span class="c"># outputs the predicted label of X given the</span>
<span class="c"># trained weights of a neural network (Theta1, Theta2)</span>
<span class="c"># Similar to feedforward process.</span>
<span class="c">##########################################################################</span>
<span class="k">function</span><span class="nf"> predict</span><span class="p">(</span><span class="n">Theta1</span><span class="p">,</span> <span class="n">Theta2</span><span class="p">,</span> <span class="n">data</span><span class="p">)</span>
    <span class="n">dataSz</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="mi">2</span><span class="p">);</span> <span class="c"># size of the data</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">dataSz</span><span class="p">,</span> <span class="mi">1</span><span class="p">);</span> <span class="c"># to save our prediction</span>
    <span class="n">h1</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">Theta1</span><span class="o">&#39;*</span><span class="p">[</span><span class="n">ones</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">dataSz</span><span class="p">),</span> <span class="n">data</span><span class="p">]);</span> <span class="c"># hidded layer output</span>
    <span class="n">h2</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">Theta2</span><span class="o">&#39;*</span><span class="p">[</span><span class="n">ones</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">size</span><span class="p">(</span><span class="n">h1</span><span class="p">,</span><span class="mi">2</span><span class="p">)),</span> <span class="n">h1</span><span class="p">]);</span> <span class="c"># output layer</span>
    <span class="c"># find the index with the max value in the array of size 10</span>
    <span class="c"># subtract 1 from the index since we are using 1 to </span>
    <span class="c"># represent 0, 2 for 1 and so on (while calculating Y)</span>
    <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">dataSz</span>
        <span class="n">p</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">=</span> <span class="n">indmax</span><span class="p">(</span><span class="n">h2</span><span class="p">[:,</span><span class="n">i</span><span class="p">])</span><span class="o">-</span><span class="mi">1</span><span class="p">;</span>
    <span class="k">end</span>
    <span class="k">return</span> <span class="n">p</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[10]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>predict (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># make prediction</span>
<span class="n">pred</span> <span class="o">=</span> <span class="n">predict</span><span class="p">(</span><span class="n">Theta1</span><span class="p">,</span> <span class="n">Theta2</span><span class="p">,</span> <span class="n">X</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">###############################################</span>
<span class="c"># calculate the accuracy of the prediction</span>
<span class="c">###############################################</span>
<span class="k">function</span><span class="nf"> accuracy</span><span class="p">(</span><span class="n">truth</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
    <span class="c"># calculate the % of predicted values</span>
    <span class="c"># matching the actual values</span>
    <span class="n">n</span> <span class="o">=</span> <span class="n">length</span><span class="p">(</span><span class="n">truth</span><span class="p">);</span>
    <span class="n">sum</span> <span class="o">=</span><span class="mi">0</span><span class="p">;</span>
    <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">n</span>
        <span class="k">if</span> <span class="n">truth</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">==</span> <span class="n">prediction</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span>
            <span class="n">sum</span> <span class="o">=</span> <span class="n">sum</span> <span class="o">+</span><span class="mi">1</span><span class="p">;</span>
        <span class="k">end</span>
    <span class="k">end</span>
  <span class="k">return</span> <span class="p">(</span><span class="n">sum</span><span class="o">/</span><span class="n">n</span><span class="p">)</span><span class="o">*</span><span class="mi">100</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[12]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>accuracy (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># calculate accuracy</span>
<span class="n">println</span><span class="p">(</span><span class="s">&quot;train accuracy: &quot;</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="n">pred</span><span class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>train accuracy: 87.01333333333334
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>After calculating the accuracy on the train dataset, let's check the accuracy on the test dataset to be sure that we did not overfit the data. If there is too much difference then we might have to tune some parameters.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ===============</span>
<span class="c"># load test data</span>
<span class="c"># ===============</span>
<span class="n">XTest</span><span class="p">,</span><span class="n">yTest</span> <span class="o">=</span> <span class="n">testdata</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># make prediction</span>
<span class="n">predTest</span> <span class="o">=</span> <span class="n">predict</span><span class="p">(</span><span class="n">Theta1</span><span class="p">,</span> <span class="n">Theta2</span><span class="p">,</span> <span class="n">XTest</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># calculate accuracy</span>
<span class="n">println</span><span class="p">(</span><span class="s">&quot;test accuracy: &quot;</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">(</span><span class="n">yTest</span><span class="p">,</span> <span class="n">predTest</span><span class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>test accuracy: 86.18
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>That's all folks! Now our model can make the prediction on any new handwritten digit in a similar way as we made the prediction on the test dataset. If we had let the training go on for longer iteration the accuracy would have been better, and there are other different ways as well to further improve the performance of the model or to make it run faster. I'll leave them for the coming posts.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">References:</h2>

<ul>
<li><a href="http://arxiv.org/abs/1404.7828">Deep Learning in Neural Networks: An Overview</a></li>
<li><a href="http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf">Learning representations by back-propagating errors</a></li>
<li><a href="http://arxiv.org/abs/1206.5533">Practical recommendations for gradient-based training of deep architectures</a></li>
<li><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">Efficient BackProp</a></li>
<li><a href="https://class.coursera.org/ml-005">Coursera Machine Learning</a></li>
<li><a href="http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html">Deep Learning</a></li>
<li><a href="http://work.caltech.edu/slides/slides10.pdf">Learning from Data</a></li>
</ul>

</div>
</div>
</div>
    </div>
  </div>
