---
layout:     post
title:      "Neural Network"
subtitle:   "digit decognition using MNIST dataset"
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
<p>Single neuron has limited computational power and hence we need a way to build a network of neurons to make a more complex model. In this post we will look into how to construct a neural network and try to solve the handwritten digit recognition problem.</p>
<p><img src="/img/nn/300px-Colored_neural_network.svg.png" alt="img src">
We'll use the <a href="http://yann.lecun.com/exdb/mnist/">MNIST dataset</a>. Luckily, <a href="https://github.com/johnmyleswhite/MNIST.jl">John Myles White</a> has already created a package to import this dataset in Julia.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">#Pkg.update();</span>
<span class="n">Pkg</span><span class="o">.</span><span class="n">clone</span><span class="p">(</span><span class="s">&quot;https://github.com/johnmyleswhite/MNIST.jl.git&quot;</span><span class="p">);</span>
<span class="n">Pkg</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="s">&quot;PyPlot&quot;</span><span class="p">)</span>
<span class="c">#Pkg.installed();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>INFO: Cloning MNIST from https://github.com/johnmyleswhite/MNIST.jl.git
INFO: Computing changes...
INFO: No packages to install, update or remove
INFO: Package database updated
INFO: Nothing to be done
INFO: METADATA is out-of-date — you may not have the latest version of PyPlot
INFO: Use &#96;Pkg.update()&#96; to get the latest versions of your packages
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
<p>For plotting, PyPlot is a good option. It provides a Julia interface to the Matplotlib plotting library from Python.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
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
<h2 class="section-heading">Activation Function</h2><p>Using a linear activation function does not give us much advantage. Linear function applied to a linear function is itself a linear function, and hence both the functions can be replaced by a single linear function. Moreover, generally speaking, real world problems are generally more complex. A linear activation function may not be a good fit for the dataset we have. Therefore if the data we wish to model is non-linear then we need to account for that in our model. Sigmoid activation function is a reasonably good non-linear activation function which we could use in our neural network.</p>
$$sigmoid(z) = 1/(1 + e^{-z})$$<p><img src="/img/nn/400px-SigmoidFunction.png" alt="sigmoid"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># computes the sigmoid of z</span>
<span class="k">function</span><span class="nf"> sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">)</span>
  <span class="n">g</span> <span class="o">=</span> <span class="mf">1.0</span> <span class="o">./</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">+</span> <span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">z</span><span class="p">));</span>
  <span class="k">return</span> <span class="n">g</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[4]:</div>


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
<h2 class="section-heading">Cost Function</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># computes the gradient of the sigmoid function evaluated at z</span>
<span class="k">function</span><span class="nf"> sigmoidGradient</span><span class="p">(</span><span class="n">z</span><span class="p">)</span>
  <span class="n">g</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">z</span><span class="p">));</span>
  <span class="k">return</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">)</span><span class="o">.*</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">z</span><span class="p">));</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[4]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>sigmoidGradient (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">#############################################################</span>
<span class="c"># computes the cost and gradient of the neural network. The</span>
<span class="c"># parameters for the neural network are &quot;unrolled&quot; into the vector</span>
<span class="c"># nn_params and need to be converted back into the weight matrices.</span>
<span class="c">#</span>
<span class="c"># The returned parameter grad should be a &quot;unrolled&quot; vector of the</span>
<span class="c"># partial derivatives of the neural network.</span>
<span class="c">#############################################################</span>
<span class="k">function</span><span class="nf"> costFunction</span><span class="p">(</span><span class="n">input_layer_size</span><span class="p">,</span> <span class="n">hidden_layer_size</span><span class="p">,</span> <span class="n">output_layer_size</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">Theta1</span><span class="p">,</span> <span class="n">Theta2</span><span class="p">,</span> <span class="n">lambda</span><span class="p">)</span>
  <span class="c"># ===================</span>
  <span class="c"># Feedforward process</span>
  <span class="c"># ===================</span>
  <span class="c"># input layer</span>
  <span class="c"># add one bias element</span>
  <span class="n">activation1</span> <span class="o">=</span> <span class="p">[</span><span class="n">ones</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="mi">1</span><span class="p">),</span><span class="mi">1</span><span class="p">)</span> <span class="n">X</span><span class="p">];</span>

  <span class="c"># hidden layer</span>
  <span class="n">z2</span> <span class="o">=</span> <span class="n">activation1</span><span class="o">*</span><span class="n">Theta1</span><span class="o">&#39;</span><span class="p">;</span>
  <span class="n">activation2</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">z2</span><span class="p">);</span>
  <span class="c"># add one bias element</span>
  <span class="n">activation2</span> <span class="o">=</span> <span class="p">[</span><span class="n">ones</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">activation2</span><span class="p">,</span><span class="mi">1</span><span class="p">),</span><span class="mi">1</span><span class="p">)</span> <span class="n">activation2</span><span class="p">];</span>

  <span class="c"># output layer</span>
  <span class="n">z3</span> <span class="o">=</span> <span class="n">activation2</span><span class="o">*</span><span class="n">Theta2</span><span class="o">&#39;</span><span class="p">;</span>
  <span class="n">activation3</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">z3</span><span class="p">);</span>
  <span class="n">h</span><span class="o">=</span><span class="n">activation3</span><span class="p">;</span>

  <span class="c"># ==========</span>
  <span class="c"># find cost</span>
  <span class="c"># ==========</span>
  <span class="n">J</span><span class="o">=</span><span class="mi">0</span><span class="p">;</span>
  <span class="n">JInter</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">m</span><span class="p">,</span><span class="mi">1</span><span class="p">);</span>
  <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">m</span>
    <span class="n">JInter</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">=</span> <span class="p">(</span><span class="o">-</span><span class="n">yInter</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span><span class="o">*</span><span class="n">log</span><span class="p">(</span><span class="n">h</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span><span class="o">&#39;</span><span class="p">))</span> <span class="o">-</span> <span class="p">((</span><span class="mi">1</span><span class="o">-</span><span class="n">yInter</span><span class="p">[</span><span class="n">i</span><span class="p">,:])</span><span class="o">*</span><span class="n">log</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">h</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span><span class="o">&#39;</span><span class="p">));</span>
  <span class="k">end</span>

  <span class="c"># regularization term</span>
  <span class="n">reg</span> <span class="o">=</span> <span class="p">(</span><span class="n">lambda</span><span class="o">/</span><span class="p">(</span><span class="mi">2</span><span class="o">*</span><span class="n">m</span><span class="p">))</span><span class="o">*</span><span class="p">(</span><span class="n">sum</span><span class="p">(</span><span class="n">sum</span><span class="p">(</span><span class="n">Theta1</span><span class="p">[:,</span><span class="mi">2</span><span class="p">:</span><span class="k">end</span><span class="p">]</span><span class="o">.^</span><span class="mi">2</span><span class="p">))</span> <span class="o">+</span> <span class="n">sum</span><span class="p">(</span><span class="n">sum</span><span class="p">(</span><span class="n">Theta2</span><span class="p">[:,</span><span class="mi">2</span><span class="p">:</span><span class="k">end</span><span class="p">]</span><span class="o">.^</span><span class="mi">2</span><span class="p">)));</span>

  <span class="n">J</span> <span class="o">=</span> <span class="p">(</span><span class="mi">1</span><span class="o">/</span><span class="n">m</span><span class="p">)</span><span class="o">*</span><span class="n">sum</span><span class="p">(</span><span class="n">JInter</span><span class="p">)</span> <span class="o">+</span> <span class="n">reg</span><span class="p">;</span>

  <span class="c"># ========================</span>
  <span class="c"># Backpropagation process</span>
  <span class="c"># ========================</span>
  <span class="n">Theta1_grad</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">Theta1</span><span class="p">));</span>
  <span class="n">Theta2_grad</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">Theta2</span><span class="p">));</span>

  <span class="n">delta3</span> <span class="o">=</span> <span class="n">activation3</span> <span class="o">-</span> <span class="n">yInter</span><span class="p">;</span>
  <span class="n">delta2</span> <span class="o">=</span> <span class="p">(</span><span class="n">delta3</span><span class="o">*</span><span class="n">Theta2</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">:</span><span class="k">end</span><span class="p">])</span><span class="o">.*</span><span class="n">sigmoidGradient</span><span class="p">(</span><span class="n">z2</span><span class="p">)</span> <span class="p">;</span>

  <span class="n">grad2</span> <span class="o">=</span> <span class="n">delta3</span><span class="o">&#39;*</span><span class="n">activation2</span><span class="p">;</span>
  <span class="n">grad1</span> <span class="o">=</span> <span class="n">delta2</span><span class="o">&#39;*</span><span class="n">activation1</span><span class="p">;</span>

  <span class="n">reg_theta1</span> <span class="o">=</span> <span class="p">((</span><span class="n">lambda</span><span class="o">/</span><span class="n">m</span><span class="p">)</span><span class="o">*</span><span class="n">Theta1</span><span class="p">);</span>
  <span class="n">reg_theta1</span><span class="p">[:,</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
  <span class="n">Theta1_grad</span> <span class="o">=</span> <span class="n">grad1</span><span class="o">/</span><span class="n">m</span> <span class="o">+</span> <span class="n">reg_theta1</span><span class="p">;</span>

  <span class="n">reg_theta2</span> <span class="o">=</span> <span class="p">((</span><span class="n">lambda</span><span class="o">/</span><span class="n">m</span><span class="p">)</span><span class="o">*</span><span class="n">Theta2</span><span class="p">);</span>
  <span class="n">reg_theta2</span><span class="p">[:,</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span>
  <span class="n">Theta2_grad</span> <span class="o">=</span> <span class="n">grad2</span><span class="o">/</span><span class="n">m</span> <span class="o">+</span> <span class="n">reg_theta2</span><span class="p">;</span>

  <span class="c"># return</span>
  <span class="k">return</span> <span class="n">J</span><span class="p">,</span> <span class="n">Theta1_grad</span><span class="p">,</span> <span class="n">Theta2_grad</span><span class="p">;</span>
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
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ===================</span>
<span class="c"># Load training data</span>
<span class="c"># ===================</span>
<span class="n">X</span><span class="p">,</span><span class="n">y</span> <span class="o">=</span> <span class="n">traindata</span><span class="p">();</span>
<span class="n">X</span><span class="o">=</span><span class="n">X</span><span class="o">&#39;</span><span class="p">;</span>
<span class="n">m</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="mi">1</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># ======================</span>
<span class="c"># Initialize parameters</span>
<span class="c"># ======================</span>
<span class="n">inputLayerSize</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="mi">2</span><span class="p">);</span>
<span class="n">hiddenLayerSize</span> <span class="o">=</span> <span class="mi">25</span><span class="p">;</span>
<span class="n">outputLayerSize</span> <span class="o">=</span> <span class="mi">10</span><span class="p">;</span>

<span class="c"># since we are doing multiclass classification</span>
<span class="n">eyeY</span> <span class="o">=</span> <span class="n">eye</span><span class="p">(</span><span class="n">outputLayerSize</span><span class="p">);</span>
<span class="n">intY</span> <span class="o">=</span> <span class="p">[</span><span class="nb">convert</span><span class="p">(</span><span class="kt">Int64</span><span class="p">,</span><span class="n">i</span><span class="p">)</span><span class="o">+</span><span class="mi">1</span> <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="n">y</span><span class="p">];</span>
<span class="n">yInter</span> <span class="o">=</span> <span class="n">Array</span><span class="p">(</span><span class="kt">Int64</span><span class="p">,</span><span class="n">length</span><span class="p">(</span><span class="n">y</span><span class="p">),</span><span class="n">outputLayerSize</span><span class="p">);</span>
<span class="k">for</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">m</span>
  <span class="n">yInter</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">=</span> <span class="n">eyeY</span><span class="p">[</span><span class="n">intY</span><span class="p">[</span><span class="n">i</span><span class="p">],:];</span>
<span class="k">end</span>

<span class="n">epsilon_init</span> <span class="o">=</span> <span class="mf">0.12</span><span class="p">;</span>
<span class="c"># including one bias neuron in input layer</span>
<span class="c"># weights for the links connecting input layer to the hidden layer</span>
<span class="n">Theta1</span> <span class="o">=</span> <span class="n">rand</span><span class="p">(</span><span class="n">hiddenLayerSize</span><span class="p">,</span> <span class="n">inputLayerSize</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span><span class="o">*</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">epsilon_init</span> <span class="o">-</span> <span class="n">epsilon_init</span><span class="p">;</span>
<span class="c"># including one bias neuron in hidden layer</span>
<span class="c"># weights for the links connecting hidden layer to the output layer</span>
<span class="n">Theta2</span> <span class="o">=</span> <span class="n">rand</span><span class="p">(</span><span class="n">outputLayerSize</span><span class="p">,</span> <span class="n">hiddenLayerSize</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span><span class="o">*</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">epsilon_init</span> <span class="o">-</span> <span class="n">epsilon_init</span><span class="p">;</span>

<span class="c"># Weight regularization parameter</span>
<span class="n">lambda</span> <span class="o">=</span> <span class="mi">3</span><span class="p">;</span>
<span class="c"># learning rate</span>
<span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.1</span><span class="p">;</span>
<span class="c"># number of iterations</span>
<span class="n">epoch</span> <span class="o">=</span> <span class="mi">1000</span><span class="p">;</span>
<span class="c"># cost per epoch</span>
<span class="n">J</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">epoch</span><span class="p">,</span><span class="mi">1</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="k">for</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">1</span><span class="p">:</span><span class="n">epoch</span>
  <span class="n">J</span><span class="p">[</span><span class="n">i</span><span class="p">,:],</span> <span class="n">Theta_grad1</span><span class="p">,</span> <span class="n">Theta_grad2</span> <span class="o">=</span> <span class="n">costFunction</span><span class="p">(</span><span class="n">inputLayerSize</span><span class="p">,</span> <span class="n">hiddenLayerSize</span><span class="p">,</span> <span class="n">outputLayerSize</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">yInter</span><span class="p">,</span> <span class="n">Theta1</span><span class="p">,</span> <span class="n">Theta2</span><span class="p">,</span> <span class="n">lambda</span><span class="p">);</span>
  <span class="n">Theta1</span> <span class="o">=</span> <span class="n">Theta1</span> <span class="o">-</span> <span class="n">alpha</span><span class="o">*</span> <span class="n">Theta_grad1</span><span class="p">;</span>
  <span class="n">Theta2</span> <span class="o">=</span> <span class="n">Theta2</span> <span class="o">-</span> <span class="n">alpha</span><span class="o">*</span> <span class="n">Theta_grad2</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
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
<img src="data:image/png;base64,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AEAZycmmzE6cWPl7e9une+qp0q23msJadnV2yxaznWHDBnN0WunNdG+/Lb3xhimvJ54odexoPkSXlCS99ppZGe7Z0/x1cbFZ+f3yS/M7O3Y0rzNkiPn7TZv2fJ100p5TKJ5+2pTg5GSzBzkx0XxIb+tWs9rcuPH+/zPYutXM/9NP5ueBoKHgIhByc3M1aNAg12PAEvL2SzzkvbdzfY84wvx59NHSRReV/95PP5lV1oqnOEyZsvffv327uZBj+XJzAsWsWab0btliim1aminGxx1nzizu2HHPc9u2lXbtMivKktkm0auXmevww80+43r1zCry1q2mgOfnm0KekCDNnm32Hn/zjfmZL780H/qbP9/83vC7Q6kAABkPSURBVH/+01wOUlXxkDdiHwUXgRCJRPgfRI+Qt198zLu6e20PP9wc1da8uTn94dZbzeN7O+pszRpzg11RkflQ3gcfmALbubNZ3X3+eVNe8/LMnz//bErv5s3m9/XsabZBXHqp9L//a/484ghTpkuFQuZIuH//e88pFc2ambL91VfmexddZArw5s1m7mXLSs9Tjqh//0E6/HCzB/nll83KePPmld/3v/9tVqGPOcbMtmOH+c9i61ZT9M84Y8+HFSXpxRfNa3fuXL3/fBF/+JAZAACeKikxH8I78kiz13fXLlNSS23bZo5fe+89s22jYUNzukWnTmbl98UXzZnEpUV2+XJTot97z2yJkEwZLikxq8H165vtFJ06mWIajZqfadzY/N5TTjGPJyVJn30mzZtntlrcdZf097+bEzSefNKcVPHZZ1K/fuY85eJi6dlnpcxM8/smTDCr2OecY4p36f8jsHGjKcj72sP844/m50pX33FoXPY1Ci4AAKhRJSVmtfbrr82JEps2mWKZnGz2I992mynIHTqY0hyJmG0eRUWmLK9ebVZhBw82R68984zZz7x9u/n9LVuafcj7uiCkVJ065jW3bzfH0C1fbkr20KFmD3SXLma1etgw8/fz50tNmkiTJpnS36+fKd8JCeV/74YNpvgXF5vi37mzKcVli/HPP5uiXtFXX5n37sOJGZyiAAAA4kYoZArjUUeZvy+7R/e008yZwmWVPdVib3JyTFncvNlsq2jXzhTQXr1M8X3/fbNyfOONZivFp5+aYvqnP5nzki+/3JyOcdFFZuX3hRfM75FMCS49Fq7070eO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"
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
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">##########################################################################</span>
<span class="c"># outputs the predicted label of X given the</span>
<span class="c"># trained weights of a neural network (Theta1, Theta2)</span>
<span class="c"># Similar to feedforward process.</span>
<span class="c">##########################################################################</span>
<span class="k">function</span><span class="nf"> predict</span><span class="p">(</span><span class="n">Theta1</span><span class="p">,</span> <span class="n">Theta2</span><span class="p">,</span> <span class="n">X</span><span class="p">)</span>
  <span class="n">m</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="mi">1</span><span class="p">);</span>
  <span class="n">num_labels</span> <span class="o">=</span> <span class="n">size</span><span class="p">(</span><span class="n">Theta2</span><span class="p">,</span> <span class="mi">1</span><span class="p">);</span>
  <span class="n">p</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span> <span class="mi">1</span><span class="p">);</span>
  <span class="n">h1</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">([</span><span class="n">ones</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span> <span class="n">X</span><span class="p">]</span> <span class="o">*</span> <span class="n">Theta1</span><span class="o">&#39;</span><span class="p">);</span>
  <span class="n">h2</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">([</span><span class="n">ones</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span> <span class="n">h1</span><span class="p">]</span> <span class="o">*</span> <span class="n">Theta2</span><span class="o">&#39;</span><span class="p">);</span>
  <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">m</span>
    <span class="c"># sub 1 from the index since we are using 1 to represent 0, 2 for 3 and so on (while calculating yInter)</span>
    <span class="n">p</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">=</span> <span class="n">indmax</span><span class="p">(</span><span class="n">h2</span><span class="p">[</span><span class="n">i</span><span class="p">,:])</span><span class="o">-</span><span class="mi">1</span><span class="p">;</span>
  <span class="k">end</span>
 <span class="k">return</span> <span class="n">p</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
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
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c">###############################################</span>
<span class="c"># calculate the accuracy of the prediction</span>
<span class="c">###############################################</span>
<span class="k">function</span><span class="nf"> accuracy</span><span class="p">(</span><span class="n">truth</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
  <span class="n">m</span> <span class="o">=</span> <span class="n">length</span><span class="p">(</span><span class="n">truth</span><span class="p">);</span>
  <span class="k">if</span> <span class="n">m</span><span class="o">!=</span><span class="n">length</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span>
    <span class="nb">error</span><span class="p">(</span><span class="s">&quot;truth and prediction length mismatch&quot;</span><span class="p">);</span>
  <span class="k">end</span>

  <span class="n">sum</span> <span class="o">=</span><span class="mi">0</span><span class="p">;</span>
  <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">m</span>
    <span class="k">if</span> <span class="n">y</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span> <span class="o">==</span> <span class="n">pred</span><span class="p">[</span><span class="n">i</span><span class="p">,:]</span>
      <span class="n">sum</span> <span class="o">=</span> <span class="n">sum</span> <span class="o">+</span><span class="mi">1</span><span class="p">;</span>
    <span class="k">end</span>
  <span class="k">end</span>
  <span class="k">return</span> <span class="p">(</span><span class="n">sum</span><span class="o">/</span><span class="n">m</span><span class="p">)</span><span class="o">*</span><span class="mi">100</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
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
<pre>train accuracy: 90.86999999999999
</pre>
</div>
</div>

<div class="output_area"><div class="prompt output_prompt">Out[18]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>90.86999999999999</pre>
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
<p>After calculating the accuracy on the train dataset, let's check the accuracy on the test dataset to be sure that we did not overfit the data.</p>

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
<span class="n">XTest</span><span class="o">=</span><span class="n">XTest</span><span class="o">&#39;</span><span class="p">;</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
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
<div class="prompt input_prompt">In&nbsp;[16]:</div>
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
<pre>test accuracy: 91.75
</pre>
</div>
</div>

</div>
</div>

</div>
    </div>
  </div>
