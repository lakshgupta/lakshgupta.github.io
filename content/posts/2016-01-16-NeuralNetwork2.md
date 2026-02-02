---
title: "Neural Network 2"
subtitle: "improve accuracy, ReLU, softmax, mini-batch gradient descent"
date: 2016-01-16T12:00:00
author: "Laksh Gupta"
header_img: "img/sd5-bg.jpg"
---
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This post is a continuation of the <a href="http://lakshgupta.github.io/2015/06/12/NeuralNetwork/">Neural Network</a> post where we learned about the basics of a neural network and applied it on the handwritten digit recognition problem. Here we'll cover the following topics which can help our neural network to perform better in terms of the accuracy of the model.</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks">Rectified linear unit (ReLU) function</a>)</li>
<li><a href="https://en.wikipedia.org/wiki/Softmax_function">Softmax function</a></li>
<li><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Mini-batch gradient descent</a></li>
</ul>
<p>So let's get started!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">using</span> <span class="n">MNIST</span>
<span class="k">using</span> <span class="n">PyPlot</span>
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
<h4 class="section-heading">ReLU Activation Function : $$f(x) = max(0, x)$$</h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="/notebooks/img/nn/sigmoid_relu.png" alt="sigmoid_relu"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We have already seen the sigmoid function instead of which we'll use ReLU activation function for the input and hidden layers in the current neural network architecture because it is faster and does not suffer from the <a href="https://en.wikipedia.org/wiki/Vanishing_gradient_problem">vanishing gradient problem</a>.</p>
<blockquote><ul>
<li>Biological plausibility: One-sided, compared to the antisymmetry of <a href="https://en.wikipedia.org/wiki/Hyperbolic_function#Tanh">tanh</a>.</li>
<li>Sparse activation: For example, in a randomly initialized network, only about 50% of hidden units are activated (having a non-zero output).</li>
<li>Efficient gradient propagation: No <a href="https://en.wikipedia.org/wiki/Vanishing_gradient_problem">vanishing gradient problem</a> or exploding effect.</li>
<li>Efficient computation: Only comparison, addition and multiplication.</li>
</ul>
<p>For the first time in 2011, the use of the rectifier as a non-linearity has been shown to enable training deep supervised neural networks without requiring unsupervised pre-training. Rectified linear units, compared to <a href="https://en.wikipedia.org/wiki/Sigmoid_function">sigmoid</a> function or similar activation functions, allow for faster and effective training of deep neural architectures on large and complex datasets.</p>
<p>Potential problems:
Non-differentiable at zero: however it is differentiable at any point arbitrarily close to 0.</p>
<p>- <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">Wikipedia</a></p><p>"What neuron type should I use?" Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of "dead" units in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work worse than ReLU/Maxout.</p>
<p>- <a href="http://cs231n.github.io/neural-networks-1/">Andrej Karpathy</a></p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ReLU function</span>
<span class="k">function</span><span class="nf"> relu</span><span class="p">(</span><span class="n">z</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="k">return</span> <span class="n">max</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="n">z</span><span class="p">);</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[2]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>relu (generic function with 1 method)</pre>
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
<div class=" highlight hl-julia"><pre><span class="c"># sigmoid function</span>
<span class="k">function</span><span class="nf"> sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="n">g</span> <span class="o">=</span> <span class="mf">1.0</span> <span class="o">./</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">+</span> <span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">z</span><span class="p">));</span>
    <span class="k">return</span> <span class="n">g</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[3]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>sigmoid (generic function with 1 method)</pre>
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
<h4 class="section-heading">Softmax Function : $$f_i(x) = \frac{e^{x_i}}{\sum_k e^{x_k}}$$</h4><p>Softmax function gives us normalized class probabilities. It takes the input, exponentiates it to generate unnormalized probabilities and then it uses a normalization factor to result in normalized probabilities. The output for each class lies between $0$ and $1$, and the sum of all the class probabilities is equal to $1$.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 class="section-heading">Forward Process</h4><p>I have re-structured the program to have replaceable components to make the experiment easy. There is a small change in the architecture of the neural network:</p>
<ul>
<li>output layer will always be using the Softmax activation function</li>
<li>rest of all the layers we'll either use ReLU activation function</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="/notebooks/img/nn/softmax_nn.jpg" alt="sigmoid"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>With the introduction of the ReLU and Softmax function we can look at our forward step. 
At the first input layer:
$$a^{(1)} = x$$</p>
<p>At the hidden layer:
$$
\begin{array}{ll}
z^{(2)} = \theta^{(1)}*a^{(1)} + bias^{(1)} \\[2ex]
a^{(2)} = ReLU(z^{(2)})
\end{array} 
$$</p>
<p>At the output layer:
$$
\begin{array}{ll}
z^{(3)} = \theta^{(2)}*a^{(2)} + bias^{(2)} \\[2ex]
\hat y = softmax(z^{(3)})
\end{array} 
$$</p>
<p>Each entry in $p$ vector defines our output normalized probability for that specific class.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">function</span><span class="nf"> forwardNN</span><span class="p">(</span><span class="n">activationFn</span><span class="p">::</span><span class="n">Function</span><span class="p">,</span> <span class="n">x</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="kd">global</span> <span class="n">network</span><span class="p">;</span>
    <span class="c"># collect input for each layer</span>
    <span class="c"># the last element will be the output from the neural network</span>
    <span class="n">activation</span> <span class="o">=</span> <span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">}[];</span>
    <span class="c"># initialize activation vector with the actual data</span>
    <span class="n">push</span><span class="o">!</span><span class="p">(</span><span class="n">activation</span><span class="p">,</span> <span class="n">x</span><span class="p">);</span>
    <span class="k">for</span> <span class="n">layer</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)</span><span class="o">-</span><span class="mi">1</span>
        <span class="n">push</span><span class="o">!</span><span class="p">(</span><span class="n">activation</span><span class="p">,</span> <span class="n">activationFn</span><span class="p">((</span><span class="n">activation</span><span class="p">[</span><span class="n">layer</span><span class="p">]</span><span class="o">*</span><span class="n">network</span><span class="p">[</span><span class="n">layer</span><span class="p">][</span><span class="mi">1</span><span class="p">])</span> <span class="o">.+</span> <span class="n">network</span><span class="p">[</span><span class="n">layer</span><span class="p">][</span><span class="mi">2</span><span class="p">]))</span>
    <span class="k">end</span>
    <span class="c"># softmax on last layer</span>
    <span class="n">score</span> <span class="o">=</span> <span class="n">activation</span><span class="p">[</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)]</span><span class="o">*</span><span class="n">network</span><span class="p">[</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)][</span><span class="mi">1</span><span class="p">]</span> <span class="o">.+</span> <span class="n">network</span><span class="p">[</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)][</span><span class="mi">2</span><span class="p">]</span>
    <span class="n">exp_scores</span> <span class="o">=</span> <span class="n">exp</span><span class="p">(</span><span class="n">score</span><span class="p">);</span>
    <span class="n">yCap</span> <span class="o">=</span> <span class="n">exp_scores</span> <span class="o">./</span> <span class="n">sum</span><span class="p">(</span><span class="n">exp_scores</span><span class="p">,</span> <span class="mi">2</span><span class="p">);</span> 

    <span class="k">return</span> <span class="n">activation</span><span class="p">,</span> <span class="n">yCap</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
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
<h4 class="section-heading">Cost Function: $J$</h4><p>The <a href="http://lakshgupta.github.io/2015/06/12/NeuralNetwork/">last time</a> we converted each output to an array of size $10$ with $1$ on the index representing the actual output and $0$ on the rest of the indices. Therefore we used a special case of the cross-entropy cost function where number of classes is equal to 2, assuming all of the output classes are independent of each other:</p>
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} \sum_{i=1}^{k}[ y^{(i)}_k\log{(h_{\theta}(x^{(i)})_k)} + (1-y^{(i)}_k)\log({1-(h_{\theta}(x^{(i)}))_k)}]$$<blockquote><p>If we have multiple independent binary attributes by which to classify the data, we can use a network with multiple logistic outputs and cross-entropy error. For multinomial classification problems (1-of-n, where n &gt; 2) we use a network with n outputs, one corresponding to each class, and target values of 1 for the correct class, and 0 otherwise. Since these targets are not independent of each other, however, it is no longer appropriate to use logistic output units. The corect generalization of the logistic sigmoid to the multinomial case is the softmax activation function.</p>
<p>- <a href="https://www.willamette.edu/~gorr/classes/cs449/classify.html">Genevieve (Jenny) B. Orr</a></p>
</blockquote>
<p>Since we are using softmax in the output layer, the probability for one class is divided by the sum of probabilities for all the classes. As a result, we will be using the generalized cross entropy cost function:</p>
\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}<blockquote><p>In the probabilistic interpretation, we are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing Maximum Likelihood Estimation (MLE).</p>
<p>- <a href="http://cs231n.github.io/linear-classify/">Andrej Karpathy</a></p>
</blockquote>
<p>More detailed information can be found <a href="http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/">here</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">function</span><span class="nf"> costFunction</span><span class="p">(</span><span class="n">truth</span><span class="p">::</span><span class="n">Vector</span><span class="p">{</span><span class="kt">Float64</span><span class="p">},</span> <span class="n">probability</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="kd">global</span> <span class="n">network</span><span class="p">;</span>
    <span class="c"># average cross-entropy loss </span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">truth</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
    
    <span class="n">corect_logprobs</span> <span class="o">=</span> <span class="p">[</span><span class="o">-</span><span class="n">log</span><span class="p">(</span><span class="n">probability</span><span class="p">[</span><span class="n">j</span><span class="p">,</span><span class="nb">convert</span><span class="p">(</span><span class="kt">Int32</span><span class="p">,</span> <span class="n">truth</span><span class="p">[</span><span class="n">j</span><span class="p">])])</span> <span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">m</span><span class="p">];</span>
    <span class="n">data_loss</span> <span class="o">=</span> <span class="n">sum</span><span class="p">(</span><span class="n">corect_logprobs</span><span class="p">)</span><span class="o">/</span><span class="n">m</span><span class="p">;</span>
    
    <span class="c">#L2 regularization</span>
    <span class="n">reg_loss</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)</span>
        <span class="n">reg_loss</span> <span class="o">=</span> <span class="n">reg_loss</span> <span class="o">+</span> <span class="mf">0.5</span><span class="o">*</span><span class="n">lambda</span><span class="o">*</span><span class="n">sum</span><span class="p">(</span><span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span><span class="o">.^</span><span class="mi">2</span><span class="p">)</span><span class="o">/</span><span class="n">m</span><span class="p">;</span>
    <span class="k">end</span>
    
    <span class="n">loss</span> <span class="o">=</span> <span class="n">data_loss</span> <span class="o">+</span> <span class="n">reg_loss</span><span class="p">;</span>
    <span class="k">return</span> <span class="n">loss</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[5]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>costFunction (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># gradient of the sigmoid function evaluated at a</span>
<span class="k">function</span><span class="nf"> sigmoidGradient</span><span class="p">(</span><span class="n">a</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
  <span class="k">return</span> <span class="n">a</span><span class="o">.*</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">a</span><span class="p">);</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># gradient of the ReLU function evaluated at a</span>
<span class="k">function</span><span class="nf"> reluGradient</span><span class="p">(</span><span class="n">a</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="n">grad</span> <span class="o">=</span> <span class="n">ones</span><span class="p">(</span><span class="n">a</span><span class="p">);</span>
    <span class="n">grad</span><span class="p">[</span><span class="n">a</span><span class="o">.&lt;=</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="k">return</span> <span class="n">grad</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[7]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>reluGradient (generic function with 1 method)</pre>
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
<h4 class="section-heading">Backward process</h4><p>The algorithm to learn the parameters is still the same, backpropagation. We need to calculate the parameter gradients to update them using the chain rule. For simplicity consider the cost function for a single input,</p>
\begin{align}
J(\theta) &= - \left[ \sum_{k=1}^{K} y_k* \log (\hat y_k) \right]
\end{align}<p>where,</p>
$$\hat y_k = softmax(z^3_k) = \frac{\exp(z^3_k)}{\sum_{j=1}^K \exp(z^3_j) }$$<p>and $y_k$ is either $0$ or $1$ as per the probability of the correct class.</p>
<p>therefore, 
$$ \frac{\partial J}{\partial \hat y_k} = − \frac{y_k}{\hat y_k} $$
$$ \frac{\partial \hat y_k}{\partial z^{(3)}_i} = \begin{cases} 
\hat y_k(1-\hat y_k),  & \text{i = k} \\\[2ex]
-\hat y_i \hat y_k, & \text{i $\\neq$ k} \\\[2ex]
\end{cases} \\\
$$</p>
$$\begin{eqnarray}
\frac{\partial J}{\partial z^{(3)}_i}
&=&\sum_{k = 1}^{K}\frac{\partial J}{\partial \hat y_k}\frac{\partial \hat y_k}{\partial z^{(3)}_i} \\\n&=& \underbrace{\frac{\partial J}{\partial \hat y_i}\frac{\partial \hat y_i}{\partial x_i}}_{i = k}
+ \underbrace{\sum_{k \neq i}\frac{\partial J(\theta)}{\partial \hat y_k}\frac{\partial \hat y_k}{\partial x_i}}_{i \neq k} \\\n&=&-y_i(1 - \hat y_i) + \sum_{k \neq i} y_k \hat y_k \\\n&=&-y_i + \sum_{k} y_k \hat y_k \\\n&=& \hat y_i - y_i \\\n\end{eqnarray}
$$<p>The correct output element in the vector $y$ is always $1$ else $0$ since we are now dealing with the normalized class probabilities.</p>
<p>For the softmax output layer:</p>
$$ 
\begin{eqnarray}
\dfrac{\partial J}{\partial \theta^{(2)}} 
&=& \dfrac{\partial J}{\partial z^{(3)}} \dfrac{\partial z^{(3)}}{\partial \theta^{(2)}} \\\n&=& (\hat y - y)* a^{(2)} 
\end{eqnarray}
$$$$ 
\begin{eqnarray}
\dfrac{\partial J}{\partial bias^{(2)}} 
&=& \dfrac{\partial J}{\partial z^{(3)}} \dfrac{\partial z^{(3)}}{\partial bias^{(2)}} \\\n&=& (\hat y - y)* 1 
\end{eqnarray}
$$<p>For the hidden layer with the ReLU activation function:</p>
$$\begin{eqnarray}
\frac{\partial J}{\partial \theta^{(1)}}
&=&\frac{\partial J}{\partial z^{(3)}}\frac{\partial z^{(3)}}{\partial g(z^{(2)})}\frac{\partial g(z^{(2)})}{\partial z^{(2)}}\frac{\partial z^{(2)}}{\partial \theta^{(1)}}\\\n&=& (\hat y - y)* \theta^{(2)} * g'(z^{(2)})*a^{(1)} \\\n\end{eqnarray}$$$$\begin{eqnarray}
\frac{\partial J}{\partial bias^{(1)}}
&=&\frac{\partial J}{\partial z^{(3)}}\frac{\partial z^{(3)}}{\partial g(z^{(2)})}\frac{\partial g(z^{(2)})}{\partial z^{(2)}}\frac{\partial z^{(2)}}{\partial bias^{(1)}}\\\n&=& (\hat y - y)* \theta^{(2)} * g'(z^{(2)})*1 \\\n\end{eqnarray}$$<p>Now we can update the weights as:</p>
$$\theta^{(l)} \leftarrow \theta^{(l)} - \alpha \dfrac{\partial J}{\partial \theta^{l}}$$<p>All the above calculations is being performed by the backwordNN and the updateThetas method below.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">function</span><span class="nf"> backwardNN</span><span class="p">(</span><span class="n">activationFnGrad</span><span class="p">::</span><span class="n">Function</span><span class="p">,</span> <span class="n">a</span><span class="p">::</span><span class="n">Vector</span><span class="p">{</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">}},</span> <span class="n">y</span><span class="p">::</span><span class="n">Vector</span><span class="p">{</span><span class="kt">Float64</span><span class="p">},</span> <span class="n">dscores</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="kd">global</span> <span class="n">network</span><span class="p">;</span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">y</span><span class="p">,</span><span class="mi">1</span><span class="p">);</span>
    <span class="n">delta</span> <span class="o">=</span> <span class="n">Array</span><span class="p">(</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">},</span> <span class="mi">1</span><span class="p">,</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">));</span>
    <span class="c"># start from the last layer to backpropagate the error</span>
    <span class="c"># compute the gradient on scores</span>
    <span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">size</span><span class="p">(</span><span class="n">dscores</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">dscores</span><span class="p">[</span><span class="n">j</span><span class="p">,</span><span class="nb">convert</span><span class="p">(</span><span class="kt">Int32</span><span class="p">,</span> <span class="n">y</span><span class="p">[</span><span class="n">j</span><span class="p">])]</span> <span class="o">-=</span> <span class="mi">1</span><span class="p">;</span>
    <span class="k">end</span>
    <span class="n">delta</span><span class="p">[</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)]</span> <span class="o">=</span> <span class="n">dscores</span><span class="o">/</span><span class="n">m</span><span class="p">;</span> <span class="c"># normalization factor</span>
    <span class="c"># backpropagate</span>
    <span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)</span><span class="o">-</span><span class="mi">1</span><span class="p">:</span><span class="o">-</span><span class="mi">1</span><span class="p">:</span><span class="mi">1</span>
        <span class="n">delta</span><span class="p">[</span><span class="n">j</span><span class="p">]</span> <span class="o">=</span> <span class="n">delta</span><span class="p">[</span><span class="n">j</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span><span class="o">*</span><span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="o">+</span><span class="mi">1</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span><span class="o">&#39;.*</span><span class="n">activationFnGrad</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="n">j</span><span class="o">+</span><span class="mi">1</span><span class="p">]);</span>
    <span class="k">end</span>
    <span class="k">return</span> <span class="n">delta</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">function</span><span class="nf"> updateThetas</span><span class="p">(</span><span class="n">a</span><span class="p">::</span><span class="n">Vector</span><span class="p">{</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">}},</span> <span class="n">delta</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">}})</span>
    <span class="kd">global</span> <span class="n">network</span><span class="p">;</span>
    <span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">length</span><span class="p">(</span><span class="n">network</span><span class="p">)</span>
        <span class="c"># update theta</span>
        <span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span> <span class="o">-</span> <span class="n">alpha</span><span class="o">*</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="n">j</span><span class="p">]</span><span class="o">&#39;*</span><span class="n">delta</span><span class="p">[</span><span class="n">j</span><span class="p">]</span> <span class="o">+</span> <span class="n">lambda</span><span class="o">*</span><span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="p">][</span><span class="mi">1</span><span class="p">])</span>
        <span class="c"># update bias</span>
        <span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="p">][</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span>  <span class="n">network</span><span class="p">[</span><span class="n">j</span><span class="p">][</span><span class="mi">2</span><span class="p">]</span> <span class="o">-</span> <span class="n">alpha</span><span class="o">*</span><span class="n">sum</span><span class="p">(</span><span class="n">delta</span><span class="p">[</span><span class="n">j</span><span class="p">],</span><span class="mi">1</span><span class="p">);</span>
    <span class="k">end</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[9]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>updateThetas (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">function</span><span class="nf"> predict</span><span class="p">(</span><span class="n">activationFn</span><span class="p">,</span> <span class="n">data</span><span class="p">::</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">})</span>
    <span class="n">activation</span><span class="p">,</span> <span class="n">probs</span> <span class="o">=</span> <span class="n">forwardNN</span><span class="p">(</span><span class="n">activationFn</span><span class="p">,</span> <span class="n">data</span><span class="p">);</span>
    <span class="n">predicted_class</span> <span class="o">=</span> <span class="p">[</span><span class="n">indmax</span><span class="p">(</span><span class="n">probs</span><span class="p">[</span><span class="n">i</span><span class="p">,:])</span> <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">size</span><span class="p">(</span><span class="n">probs</span><span class="p">,</span><span class="mi">1</span><span class="p">)]</span>
    <span class="k">return</span> <span class="n">predicted_class</span><span class="p">;</span>
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
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">function</span><span class="nf"> accuracy</span><span class="p">(</span><span class="n">truth</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
    <span class="n">correct</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
    <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">length</span><span class="p">(</span><span class="n">truth</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">truth</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">==</span> <span class="n">prediction</span><span class="p">[</span><span class="n">i</span><span class="p">]</span>
            <span class="n">correct</span> <span class="o">=</span> <span class="n">correct</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
        <span class="k">end</span>
    <span class="k">end</span>
    <span class="n">println</span><span class="p">(</span><span class="s">"training accuracy: "</span><span class="p">,</span> <span class="n">correct</span><span class="o">/</span><span class="n">length</span><span class="p">(</span><span class="n">truth</span><span class="p">)</span><span class="o">*</span><span class="mi">100</span><span class="p">);</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[11]:</div>


</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Training a model</h2><p>Based on the new structure of the program, we'll build up our neural network and train our model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 class="section-heading">Load Data</h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ===================</span>
<span class="c"># load training data</span>
<span class="c"># ===================</span>
<span class="n">X</span><span class="p">,</span><span class="n">Y</span> <span class="o">=</span> <span class="n">traindata</span><span class="p">();</span> <span class="c">#X:(784x60000), y:(60000x1)</span>
<span class="n">X</span> <span class="o">/=</span> <span class="mf">255.0</span><span class="p">;</span> <span class="c"># scale the input between 0 and 1</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">X</span><span class="o">'</span><span class="p">;</span> <span class="c">#X:(60000X784)</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">Y</span><span class="o">+</span><span class="mi">1</span><span class="p">;</span> <span class="c"># adding 1 to handle index, hence value 1 represent digit 0 now</span>
<span class="c"># number of instances</span>
<span class="n">println</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span><span class="mi">1</span><span class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>60000
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
<h4 class="section-heading">Define Network</h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="n">inputLayerSize</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="mi">2</span><span class="p">);</span>
<span class="n">hiddenLayerSize</span> <span class="o">=</span> <span class="mi">100</span><span class="p">;</span>
<span class="n">outputLayerSize</span> <span class="o">=</span> <span class="mi">10</span><span class="p">;</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># define a network</span>
<span class="n">network</span> <span class="o">=</span> <span class="p">[];</span>

<span class="c"># add first layer to the network</span>
<span class="n">layer1</span> <span class="o">=</span> <span class="n">Array</span><span class="p">(</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">},</span> <span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="c">#theta1</span>
<span class="n">layer1</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mf">0.01</span><span class="o">*</span><span class="n">randn</span><span class="p">(</span><span class="n">inputLayerSize</span><span class="p">,</span> <span class="n">hiddenLayerSize</span><span class="p">);</span> 
<span class="c">#bias1</span>
<span class="n">layer1</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">hiddenLayerSize</span><span class="p">);</span> 
<span class="n">push</span><span class="o">!</span><span class="p">(</span><span class="n">network</span><span class="p">,</span><span class="n">layer1</span><span class="p">);</span>

<span class="c"># add second layer to the network</span>
<span class="n">layer2</span> <span class="o">=</span> <span class="n">Array</span><span class="p">(</span><span class="n">Matrix</span><span class="p">{</span><span class="kt">Float64</span><span class="p">},</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="c">#theta2</span>
<span class="n">layer2</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mf">0.01</span><span class="o">*</span><span class="n">randn</span><span class="p">(</span><span class="n">hiddenLayerSize</span><span class="p">,</span> <span class="n">outputLayerSize</span><span class="p">);</span> 
<span class="c">#bias2</span>
<span class="n">layer2</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">outputLayerSize</span><span class="p">);</span> 
<span class="n">push</span><span class="o">!</span><span class="p">(</span><span class="n">network</span><span class="p">,</span><span class="n">layer2</span><span class="p">);</span>
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
<h4 class="section-heading">Training: Mini-batch gradient descent</h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The last time we used gradient descent to train our model. In this implementation we'll switch to Mini-Batch gradient descent algorithm which is not much different from regular gradient descent. It's just that instead of working on the whole dataset we'll work on a smaller dataset in each iteration. Because of this change our training algorithm will become computationally faster since large datasets are difficult to handle in memory which makes vectorization much less efficient.</p>
<blockquote><p>In particular, suppose that our error function is particularly pernicious and has a bunch of little valleys. If we used the entire training set to compute each gradient, our model would get stuck in the first valley it fell into (since it would register a gradient of 0 at this point). If we use smaller mini-batches, on the other hand, we'll get more noise in our estimate of the gradient. This noise might be enough to push us out of some of the shallow valleys in the error function.</p>
<p>- <a href="https://www.quora.com/Intuitively-how-does-mini-batch-size-affect-the-performance-of-gradient-descent">Quora</a></p>
</blockquote>
<p>One thing to take care in the while training is that mini-batches need to be balanced for classes otherwise the estimation of the gradient using mini-batch gradient descent would be way off then the gradient calculated using the whole dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="n">alpha</span> <span class="o">=</span> <span class="mf">1e-0</span><span class="p">;</span> <span class="c"># step size</span>
<span class="n">lambda</span> <span class="o">=</span> <span class="mf">1e-3</span><span class="p">;</span> <span class="c"># regularization factor</span>
<span class="n">numItr</span> <span class="o">=</span> <span class="mi">1500</span><span class="p">;</span>
<span class="n">sampleCostAtEveryItr</span> <span class="o">=</span> <span class="mi">10</span><span class="p">;</span>
<span class="n">batchSize</span> <span class="o">=</span> <span class="nb">convert</span><span class="p">(</span><span class="kt">Int32</span><span class="p">,</span><span class="n">floor</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span><span class="o">/</span><span class="mi">10</span><span class="p">));</span>
<span class="n">J</span> <span class="o">=</span> <span class="p">[];</span>
<span class="k">for</span> <span class="n">itr</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">numItr</span>
    <span class="c"># take next batch of random instances </span>
    <span class="n">minibatch</span> <span class="o">=</span> <span class="n">collect</span><span class="p">(</span><span class="n">rand</span><span class="p">(</span><span class="mi">1</span><span class="p">:</span><span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="mi">1</span><span class="p">),</span> <span class="n">batchSize</span><span class="p">));</span>
    <span class="c"># feedforward</span>
    <span class="n">activations</span><span class="p">,</span> <span class="n">probs</span> <span class="o">=</span> <span class="n">forwardNN</span><span class="p">(</span><span class="n">relu</span><span class="p">,</span> <span class="n">X</span><span class="p">[</span><span class="n">minibatch</span><span class="p">,:]);</span> 
    <span class="c"># cost</span>
    <span class="k">if</span> <span class="n">itr</span><span class="o">%</span><span class="n">sampleCostAtEveryItr</span> <span class="o">==</span> <span class="mi">1</span>
        <span class="n">activationsX</span><span class="p">,</span> <span class="n">probsX</span> <span class="o">=</span> <span class="n">forwardNN</span><span class="p">(</span><span class="n">relu</span><span class="p">,</span> <span class="n">X</span><span class="p">);</span> 
        <span class="n">push</span><span class="o">!</span><span class="p">(</span><span class="n">J</span><span class="p">,</span> <span class="n">costFunction</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span> <span class="n">probsX</span><span class="p">));</span>
    <span class="k">end</span>
    <span class="c"># backpropagation</span>
    <span class="n">newThetas</span> <span class="o">=</span> <span class="n">backwardNN</span><span class="p">(</span><span class="n">reluGradient</span><span class="p">,</span> <span class="n">activations</span><span class="p">,</span> <span class="n">Y</span><span class="p">[</span><span class="n">minibatch</span><span class="p">],</span> <span class="n">probs</span><span class="p">);</span>
    <span class="c"># update parameters</span>
    <span class="n">updateThetas</span><span class="p">(</span><span class="n">activations</span><span class="p">,</span> <span class="n">newThetas</span><span class="p">);</span>
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
<p>The code above is using ReLU activation function for the hidden layers but the program is modular enough to let us experiment with different activation functions, for example sigmoid.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Prediction and Accuracy</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="n">accuracy</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span> <span class="n">predict</span><span class="p">(</span><span class="n">relu</span><span class="p">,</span> <span class="n">X</span><span class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>training accuracy: 97.975
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
<p>This is approximately 10% improvement over our previous implementation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># plot the cost per iteration</span>
<span class="n">sampleIdxJ</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="o">+</span><span class="n">sampleCostAtEveryItr</span><span class="o">*</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">0</span><span class="p">:</span><span class="n">length</span><span class="p">(</span><span class="n">J</span><span class="p">)</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
<span class="n">plot</span><span class="p">(</span><span class="n">sampleIdxJ</span><span class="p">,</span> <span class="n">J</span><span class="p">)</span>
<span class="n">xlabel</span><span class="p">(</span><span class="s">"Sampled Iterations"</span><span class="p">)</span>
<span class="n">ylabel</span><span class="p">(</span><span class="s">"Cost"</span><span class="p">)</span>
<span class="n">grid</span><span class="p">(</span><span class="s">"on"</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAsoAAAItCAYAAAAg11x6AAAABHNCSVQICAgIfAhkiAAAA... (truncated) ..." >
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The spikes in the graph are due to the use of mini-batch gradient descent, which estimates the cost over the whole dataset hence sometimes moves away from the minima but in the end converges satisfactorily.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># test data</span>
<span class="n">XTest</span><span class="p">,</span><span class="n">YTest</span> <span class="o">=</span> <span class="n">testdata</span><span class="p">();</span>
<span class="n">XTest</span> <span class="o">/=</span> <span class="mf">255.0</span><span class="p">;</span>
<span class="n">XTest</span> <span class="o">=</span> <span class="n">XTest</span><span class="o">'</span><span class="p">;</span>
<span class="n">YTest</span> <span class="o">=</span> <span class="n">YTest</span><span class="o">+</span><span class="mi">1</span><span class="p">;</span>
<span class="n">accuracy</span><span class="p">(</span><span class="n">YTest</span><span class="p">,</span> <span class="n">predict</span><span class="p">(</span><span class="n">relu</span><span class="p">,</span> <span class="n">XTest</span><span class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>training accuracy: 97.24000000000001
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
<h2 class="section-heading">References:</h2><ul>
<li><a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks">Rectified linear unit (ReLU)</a>)</li>
<li><a href="http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf">Learning representations by back-propagating errors</a></li>
<li><a href="http://arxiv.org/abs/1206.5533">Practical recommendations for gradient-based training of deep architectures</a></li>
<li><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">Efficient BackProp</a></li>
<li><a href="http://www.iro.umontreal.ca/~bengioy/dlbook/mlp.html">Deep Learning</a></li>
<li><a href="http://arxiv.org/abs/1411.2738">word2vec Parameter Learning Explained</a></li>
<li><a href="https://www.reddit.com/r/cs231n/comments/45u13l/binary_cross_entropy_cost_function_with_softmax/">binary cross entropy cost function with softmax?</a></li>
<li><a href="http://arxiv.org/abs/cond-mat/0512017">Combinatorial Information Theory: I. Philosophical Basis of Cross-Entropy and Entropy</a></li>
<li><a href="https://courses.cs.ut.ee/MTAT.03.277/2015_fall/uploads/Main/word2vec.pdf">word2vec gradients</a></li>
<li><a href="http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/">Softmax Regression</a></li>
</ul>

</div>
</div>
</div>
    </div>
  </div>

```