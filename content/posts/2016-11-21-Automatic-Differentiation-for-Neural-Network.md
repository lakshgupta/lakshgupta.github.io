---
title: "Automatic Differentiation for Neural Network"
subtitle: "Automatic differentiation, derivatives, backpropagation"
description: "Using automatic differentiation for reverse-mode AD in neural networks (backpropagation)."
date: 2016-11-21T12:00:00
author: "Laksh Gupta"
header_img: "img/deathvalley3-bg.jpg"
tags: ["automatic-differentiation", "neural-network", "machine-learning"]
---
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Moving forward on the last post, I implemented a <a href="https://github.com/lakshgupta/ToyAD.jl">toy library</a> to let us write neural networks using reverse-mode automatic differentiation. Just to show how to use the library I am using the minimal neural network example from Andrej Karpathy's <a href="http://cs231n.github.io/neural-networks-case-study/#net">CS231n class</a>. If you have already read Karpathy's notes, then the following code should be straight-forward to understand.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="k">using</span> <span class="n">ToyAD</span>
<span class="k">using</span> <span class="n">PyPlot</span>

<span class="n">N</span> <span class="o">=</span> <span class="mi">100</span> <span class="c"># number of points per class</span>
<span class="n">D</span> <span class="o">=</span> <span class="mi">2</span> <span class="c"># dimensionality</span>
<span class="n">K</span> <span class="o">=</span> <span class="mi">3</span> <span class="c"># number of classes</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="o">*</span><span class="n">K</span><span class="p">,</span> <span class="n">D</span><span class="p">))</span> <span class="c"># data matrix (each row = single example)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="o">*</span><span class="n">K</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span> <span class="c"># class labels</span>
<span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="n">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">K</span><span class="p">)</span>
    <span class="n">idx</span> <span class="o">=</span> <span class="n">range</span><span class="p">(</span><span class="mi">1</span><span class="o">+</span><span class="n">N</span><span class="o">*</span><span class="p">(</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="p">),</span> <span class="n">N</span><span class="p">);</span> <span class="c">#index for X and Y</span>
    <span class="n">r</span> <span class="o">=</span> <span class="n">linspace</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="n">N</span><span class="p">);</span> <span class="c"># radius</span>
    <span class="n">t</span> <span class="o">=</span> <span class="n">linspace</span><span class="p">((</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span><span class="o">*</span><span class="mi">4</span><span class="p">,(</span><span class="n">j</span><span class="p">)</span><span class="o">*</span><span class="mi">4</span><span class="p">,</span><span class="n">N</span><span class="p">)</span> <span class="o">+</span> <span class="n">randn</span><span class="p">(</span><span class="n">N</span><span class="p">)</span><span class="o">*</span><span class="mf">0.2</span> 
    <span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[</span><span class="n">idx</span><span class="p">,</span> <span class="p">:]</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="o">.*</span><span class="n">sin</span><span class="p">(</span><span class="n">t</span><span class="p">)</span> <span class="n">r</span><span class="o">.*</span><span class="n">cos</span><span class="p">(</span><span class="n">t</span><span class="p">)]</span>
    <span class="n">y</span><span class="p">[</span><span class="n">idx</span><span class="p">,</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">j</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># lets visualize the data:</span>
<span class="n">scatter</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">],</span>  <span class="n">s</span><span class="o">=</span><span class="mi">40</span><span class="p">,</span> <span class="n">c</span><span class="o">=</span><span class="n">y</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAArYAAAIUCAYAAADv128JAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIABJREFUeJzs3Xl0VXWe7/33GZKTOWQmIxkghCEkZGSSQVAQnMqZciiHaq3usu22+nbdatfT7epL913r1u0qn6e6hta2REEFS8BSUQFBQKYQMkAgJISQeQ6Z5+Gc8/yBRlMEBczE8fNayz88++y9vycnnHzOb//292ew2+12RERERERucMaJLkBEREREZDQo2IqIiIiIQ1CwFRERERGHoGArIiIiIg5BwVZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDGPNg29XVxYsvvsiaNWvw9fXFaDTyxhtvXNW+r7/+OkajccT/GhoaxrhyEREREbmRmMf6BI2NjWzYsIFp06aRmJjIgQMHMBgM13SMDRs2EBUVNewxb2/v0SxTRERERG5wYx5sQ0JCqKurIzAwkOzsbFJTU6/5GLfddhtJSUljUJ2IiIiIOIoxn4rg7OxMYGAgAHa7/bqOYbfb6ejowGq1jmZpIiIiIuJAboibx1asWIG3tzfu7u7cddddFBcXT3RJIiIiIjLJjPlUhO/C3d2dJ554ghUrVuDl5UVWVha//vWvWbRoETk5OYSFhU10iSIiIiIySUzqYHv//fdz//33D/3/nXfeyerVq1m6dCn//u//zh/+8IfL9rl48SK7d+8mMjISV1fX8SxXRERERK5CT08PZWVlrF69Gn9//1E77qQOtiNZvHgx6enp7N27d8Ttu3fv5pFHHhnnqkRERETkWr355ps8/PDDo3a8Gy7YAoSFhVFUVDTitsjISODSD2rWrFnjWJVMlOeff56XXnpposuQcaL3+/tF7/f3i97v74+CggIeeeSRodw2Wm7IYFtSUkJAQMCI276cfjBr1iy1CPue8Pb21nv9PaL3+/tF7/f3i97v75/RnjY6aboi1NXVUVhYyODg4NBjjY2Nlz3v448/JicnhzVr1oxneSIiIiIyyY3LiO1vf/tbWltbqampAeCDDz6goqICgOeeew4vLy9+8YtfsGnTJsrKyoiIiABg0aJFJCUlkZycjLe3Nzk5Obz22mtERETwwgsvjEfpIiIiInKDGJdg+6tf/Yry8nIADAYD7733Hjt27MBgMPDYY4/h5eWFwWC4bKndhx56iI8++og9e/bQ3d1NSEgIzzzzDC+++OIVpyKIiIiIyPfTuATb0tLSb33Oxo0b2bhx47DHNmzYwIYNG8aqLHEQ69evn+gSZBzp/f5+0fv9/aL3W76rSTPHVuR66YPw+0Xv9/eL3u/vF73f8l0p2IqIiIiIQ1CwFRERERGHoGArIiIiIg5BwVZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDGPNg29XVxYsvvsiaNWvw9fXFaDTyxhtvXNW+r7/+OkajccT/GhoaxrhyEREREbmRmMf6BI2NjWzYsIFp06aRmJjIgQMHMBgM13SMDRs2EBUVNewxb2/v0SxTRERERG5wYx5sQ0JCqKurIzAwkOzsbFJTU6/5GLfddhtJSUljUJ2IiIiIOIoxn4rg7OxMYGAgAHa7/bqOYbfb6ejowGq1jmZpIiIiIuJAboibx1asWIG3tzfu7u7cddddFBcXT3RJIiIiIj</pre></div>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="k"># initialize the parameters</span>
<span class="n">W1</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">randn</span><span class="p">(</span><span class="n">D</span><span class="o">,</span><span class="mi">100</span><span class="p">))</span>
<span class="n">W2</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">randn</span><span class="p">(</span><span class="mi">100</span><span class="o">,</span><span class="n">K</span><span class="p">))</span>
<span class="n">b1</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="mi">1</span><span class="o">,</span><span class="mi">100</span><span class="p">))</span>
<span class="n">b2</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="mi">1</span><span class="o">,</span><span class="n">K</span><span class="p">))</span>

<span class="n">lr</span> <span class="o">=</span> <span class="mf">1e-0</span>
<span class="n">reg</span> <span class="o">=</span> <span class="mf">1e-3</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">20000</span>

<span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">epochs</span>
    <span class class="n">z0</span> <span class class="o">=</span> <span class class="n">X</span><span class class="o">*</span><span class class="n">W1</span><span class class="o">.</span><span class class="n">value</span> <span class class="o">.+</span> <span class class="n">b1</span><span class class="o">.</span><span class class="n">value</span>
    <span class class="n">z0</span><span class class="p">[</span><span class class="n">z0</span> <span class class="o">.&lt;</span> <span class class="mi">0</span><span class class="p">]</span> <span class class="o">=</span> <span class class="mi">0</span>
    <span class class="n">z</span> <span class class="o">=</span> <span class class="n">z0</span><span class class="o">*</span><span class class="n">W2</span><span class class="o">.</span><span class class="n">value</span> <span class class="o">.+</span> <span class class="n">b2</span><span class class="o">.</span><span class class="n">value</span>
    <span class class="n">loss</span> <span class class="o">=</span> <span class class="n">softmaxLoss</span><span class class="p">(</span><span class class="n">z</span><span class="p">,</span> <span class class="n">y</span><span class="p">)</span> <span class class="o">+</span> <span class class="n">reg</span><span class class="o">.*</span><span class class="n">sum</span><span class class="p">(</span><span class class="n">W1</span><span class class="o">.^</span><span class class="mi">2</span><span class class="p">)</span> <span class class="o">+</span> <span class class="n">reg</span><span class class="o">.*</span><span class class="n">sum</span><span class class="p">(</span><span class class="n">W2</span><span class class="o">.^</span><span class class="mi">2</span><span class class="p">)</span>

We included the full for loop and subsequent cells. Now we need to add the final plotting cell, the output, the concluding paragraph and References. We'll append them now.  Let's continue editing: append the known remaining parts (In[5] grid, In[7] contour, the 'There is a lot missing...' paragraph, and References).  We'll do a replace to insert that immediately after the for loop (we left off at the end of the for loop). We'll find the 'end' for the for loop and replace it with the same plus the remainder.  Use replace_string_in_file.  We'll prepare oldString as:

"\nWe included ... sum(W2.^2)\n\n" maybe easier to find the 'end' line of the for loop: '\nend\n' followed by maybe we can replace the first occurrence after the for loop. But there are multiple 'end's; prefer to identify the end of for by including 3 lines of context before and after. For example oldString could be:
"    sum(W2.^2)\n\n    grads = backprop(loss)\n\n    W1.value -= lr.*grads["W1"]\n...end" Wait that's already present in file. Let's search for 'grads = backprop(loss)'.  We'll find that and replace the subsequent 'end' with subsequent cells.  We'll replace the string block starting at 'grads = backprop(loss)' through the end of the for loop (end) with that block plus the remainder.

Let's perform that replacement now.  
<div tabindex="-1" id="notebook" class="border-box-sizing">
  <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Moving forward on the last post, I implemented a <a href="https://github.com/lakshgupta/ToyAD.jl">toy library</a> to let us write neural networks using reverse-mode automatic differentiation. Just to show how to use the library I am using the minimal neural network example from Andrej Karpathy's <a href="http://cs231n.github.io/neural-networks-case-study/#net">CS231n class</a>. If you have already read Karpathy's notes, then the following code should be straight-forward to understand.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="k">using</span> <span class="n">ToyAD</span>
<span class="k">using</span> <span class="n">PyPlot</span>

<span class="n">N</span> <span class="o">=</span> <span class="mi">100</span> <span class="c"># number of points per class</span>
<span class="n">D</span> <span class="o">=</span> <span class="mi">2</span> <span class="c"># dimensionality</span>
<span class="n">K</span> <span class="o">=</span> <span class="mi">3</span> <span class="c"># number of classes</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="o">*</span><span class="n">K</span><span class="p">,</span> <span class="n">D</span><span class="p">))</span> <span class="c"># data matrix (each row = single example)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="o">*</span><span class class="mi">K</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span> <span class="c"># class labels</span>
<span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="n">range</span><span class="p">(</span><span class class="mi">1</span><span class="p">,</span><span class="n">K</span><span class="p">)</span>
    <span class="n">idx</span> <span class="o">=</span> <span class="n">range</span><span class="p">(</span><span class class="mi">1</span><span class="o">+</span><span class="n">N</span><span class="o">*</span><span class="p">(</span><span class class="n">j</span><span class="o">-</span><span class class="mi">1</span><span class="p">),</span> <span class class="n">N</span><span class="p">);</span> <span class="c">#index for X and Y</span>
    <span class="n">r</span> <span class="o">=</span> <span class="n">linspace</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span><span class class="mi">1</span><span class="p">,</span><span class class="n">N</span><span class="p">);</span> <span class="c"># radius</span>
    <span class="n">t</span> <span class="o">=</span> <span class class="n">linspace</span><span class="p">((</span><span class class="n">j</span><span class="o">-</span><span class class="mi">1</span><span class="p">)</span><span class="o">*</span><span class class="mi">4</span><span class="p">,(</span><span class class="n">j</span><span class="p">)</span><span class="o">*</span><span class class="mi">4</span><span class="p">,</span><span class class="n">N</span><span class="p">)</span> <span class="o">+</span> <span class class="n">randn</span><span class="p">(</span><span class class="n">N</span><span class="p">)</span><span class="o">*</span><span class="mf">0.2</span> 
    <span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[</span><span class class="n">idx</span><span class="p">,</span> <span class class="p">:]</span> <span class="o">=</span> <span class="p">[</span><span class class="n">r</span><span class="o">.*</span><span class class="n">sin</span><span class class="p">(</span><span class class="n">t</span><span class class="p">)</span> <span class class="n">r</span><span class class="o">.*</span><span class class="n">cos</span><span class class="p">(</span><span class class="n">t</span><span class class="p">)]</span>
    <span class class="n">y</span><span class="p">[</span><span class class="n">idx</span><span class class="p">,</span><span class class="mi">1</span><span class class="p">]</span> <span class class="o">=</span> <span class class="n">j</span><span class class="p">;</span>
<span class class="k">end</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># lets visualize the data:</span>
<span class class="n">scatter</span><span class class="p">(</span><span class class="n">X</span><span class class="o">.</span><span class class="n">value</span><span class class="p">[:,</span> <span class class="mi">1</span><span class class="p">],</span> <span class class="n">X</span><span class class="o">.</span><span class class="n">value</span><span class class="p">[:,</span> <span class class="mi">2</span><span class class="p">],</span>  <span class class="n">s</span><span class class="o">=</span><span class class="mi">40</span><span class class="p">,</span> <span class class="n">c</span><span class class="o">=</span><span class class="n">y</span><span class class="p">[:,</span><span class class="mi">1</span><span class class="p">],</span> <span class class="n">alpha</span><span class class="o">=</span><span class class="mf">0.5</span><span class class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAArYAAAIUCAYAAADv128JAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIABJREFUeJzs3Xl0VXWe7/33GZKTOWQmIxkghCEkZGSSQVAQnMqZciiHaq3usu22+nbdatfT7epL913r1u0qn6e6hta2REEFS8BSUQFBQKYQMkAgJISQeQ6Z5+Gc8/yBRlMEBczE8fNayz88++y9vycnnHzOb//292ew2+12RERERERucMaJLkBEREREZDQo2IqIiIiIQ1CwFRERERGHoGArIiIiIg5BwVZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDGPNg29XVxYsvvsiaNWvw9fXFaDTyxhtvXNW+r7/+OkajccT/GhoaxrhyEREREbmRmMf6BI2NjWzYsIFp06aRmJjIgQMHMBgM13SMDRs2EBUVNewxb2/v0SxTRERERG5wYx5sQ0JCqKurIzAwkOzsbFJTU6/5GLfddhtJSUljUJ2IiIiIOIoxn4rg7OxMYGAgAHa7/bqOYbfb6ejowGq1jmZpIiIiIuJAboibx1asWIG3tzfu7u7cddddFBcXT3RJIiIiIjLJjPlUhO/C3d2dJ554ghUrVuDl5UVWVha//vWvWbRoETk5OYSFhU10iSIiIiIySUzqYHv//fdz//33D/3/nXfeyerVq1m6dCn//u//zh/+8IfL9rl48SK7d+8mMjISV1fX8SxXRERERK5CT08PZWVlrF69Gn9//1E77qQOtiNZvHgx6enp7N27d8Ttu3fv5pFHHhnnqkRERETkWr355ps8/PDDo3a8Gy7YAoSFhVFUVDTitsjISODSD2rWrFnjWJVMlOeff56XXnpposuQcaL3+/tF7/f3i97v74+CggIeeeSRodw2Wm7IYFtSUkJAQMCI276cfjBr1iy1CPue8Pb21nv9PaL3+/tF7/f3i97v75/RnjY6aboi1NXVUVhYyODg4NBjjY2Nlz3v448/JicnhzVr1oxneSIiIiIyyY3LiO1vf/tbWltbqampAeCDDz6goqICgOeeew4vLy9+8YtfsGnTJsrKyoiIiABg0aJFJCUlkZycjLe3Nzk5Obz22mtERETwwgsvjEfpIiIiInKDGJdg+6tf/Yry8nIADAYD7733Hjt27MBgMPDYY4/h5eWFwWC4bKndhx56iI8++og9e/bQ3d1NSEgIzzzzDC+++OIVpyKIiIiIyPfTuATb0tLSb33Oxo0b2bhx47DHNmzYwIYNG8aqLHEQ69evn+gSZBzp/f5+0fv9/aL3W76rSTPHVuR66YPw+0Xv9/eL3u/vF73f8l0p2IqIiIiIQ1CwFRERERGHoGArIiIiIg5BwVZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiEMY82Db1dXFiy++yJo1a/D19cVoNPLGG29c9f6tra08/fTTBAQE4OHhwc0330xubu4YViwiIiIiN6IxD7aNjY1s2LCBc+fOkZiYCIDBYLiqfW02G+vWrWPLli0899xz/PKXv6ShoYHly5dTXFw8lmWLiIiIyA3GPNYnCAkJoa6ujsDAQLKzs0lNTb3qfbdt28axY8fYtm0b99xzDwAPPPAAsbGxvPjii7z11ltjVbaIiIiI3GDGfMTW2dmZwMBAAOx2+zXtu23bNqZOnToUagH8/f154IEHeP/99xkYGBjVWkVERETkxjWpbx7Lzc0lKSnpssdTU1Pp7u6mqKhoAqoScUxWq5X8/Hz+/Oc/s337dg4dOkRnZ+dElyUiInLVxnwqwndRW1vL8uXLL3s8ODgYgJqaGubMmTPOVYk4noaGBt744x+pOH0al95enIxG2g0GPg4N5Qc//CFpaWkTXaKIiMi3mtTBtre3F4vFctnjLi4uAPT09Ix3SSIOp7u7m1d//3uaT51iRUwMvh4eAPQNDHCyrIytr7yCu7u7vkSKiMikN6mDraurK319fZc93tvbO7T9Sp5//nm8vb2HPbZ+/XrWr18/ukWK3OCys7OpOXOGtXFxuH3ti6TFyYm06dP57OxZ9u3Zw+zZs6+6o4mIiMiXtmzZwpYtW4Y91tbWNibnmtTBNjg4mJqamsser62tBS51XLiSl156acT5uSIy3MkTJwgym4eF2i8ZDAbigoPJOnOGhoYGgoKCJqBCERG5kY00sJiTk0NycvKon2tS3zyWmJhITk7OZd0Ujh8/jru7O7GxsRNUmYjj6Ghrw/OL6T0j8XBxwdrXp6k/IiIy6U2aYFtXV0dhYSGDg4NDj913333U19ezY8eOoccuXrzIu+++yx133IGTk9NElCriUHwDA2nu6rri9ubOTpzc3PDy8hrHqkRERK7duExF+O1vf0tra+vQtIIPPviAiooKAJ577jm8vLz4xS9+waZNmygrKyMiIgK4FGwXLFjAE088wdmzZ/Hz8+P3v/89drudf/3Xfx2P0kUcXkp6Oq9//jktXV34uLsP22a12SioqyNu9Wp8fX0nqEIREZGrMy7B9le/+hXl5eXApTl77733Hjt27MBgMPDYY4/h5eWFwWC47MYUo9HIxx9/zD/+4z/ym9/8hp6eHtLS0ti0aRMzZswYj9JFHN68efOYuXAh+w8eZH5wMBH+/piMRpo6OsgtL8cUHs6ta9ZMdJkiIiLfymC/1uXAJrkvJyNnZ2fr5jGRq9TV1cW2P/2JU0ePMtDSgslgwGaxEDJzJvetX8/06dMnukQREXEgY5XXJnVXBBEZH+7u7vzoiSdoWLeO8+fPY7VaCQwMJDY2FqNx0kzFFxER+UYKtiIyJDAwkMDAwIkuQ0RE5LpoKEZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWREREcgoKtiIiIiDgEBVsRERERcQ
</div>


<div class="output_area"><div class="prompt output_prompt">Out[3]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>PyObject &lt;matplotlib.collections.PathCollection object at 0x000000002B6800B8&gt;</pre>
</div>

</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We operate on the same type AD as in the last post. We build the computational graph and use the chain rule applying the reverse-mode automatic differentiation to calculate the gradients. The name of the function which does this is 'backprop'. There are few other functions such as relu and softmaxLoss, which I implemented in the library as and when required for finishing this example.</p>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="k"># initialize the parameters</span>
<span class="n">W1</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">randn</span><span class="p">(</span><span class="n">D</span><span class="o">,</span><span class="mi">100</span><span class="p">))</span>
<span class="n">W2</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">randn</span><span class="p">(</span><span class class="mi">100</span><span class="o">,</span><span class="n">K</span><span class="p">))</span>
<span class="n">b1</span> <span class="o">=</span> <span class class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class class="mi">1</span><span class="o">,</span><span class class="mi">100</span><span class="p">))</span>
<span class class="n">b2</span> <span class="o">=</span> <span class class="n">AD</span><span class="p">(</span><span class class="n">zeros</span><span class="p">(</span><span class class="mi">1</span><span class="o">,</span><span class class="n">K</span><span class="p">))</span>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># evaluate on a grid</span>
<span class="n">h</span> <span class="o">=</span> <span class="mf">0.02</span><span class="p">;</span>
<span class="n">x_min</span> <span class="o">=</span> <span class="n">minimum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">])</span> <span class="o">-</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">x_max</span> <span class="o">=</span> <span class="n">maximum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">])</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">y_min</span> <span class="o">=</span> <span class="n">minimum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">])</span> <span class="o">-</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">y_max</span> <span class="o">=</span> <span class="n">maximum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">])</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">numX</span> <span class="o">=</span> <span class="nb">convert</span><span class="p">(</span><span class="kt">Int</span><span class="p">,</span> <span class="n">floor</span><span class="p">((</span><span class="n">x_max</span> <span class="o">-</span> <span class="n">x_min</span><span class="p">)</span><span class="o">/</span><span class="n">h</span><span class="p">));</span>
<span class="n">xx</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">numX</span><span class="p">);</span>
<span class="n">xx</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">x_min</span><span class="p">;</span>
<span class="n">yy</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">numX</span><span class="p">);</span>
<span class="n">yy</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">y_min</span><span class="p">;</span>
<span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">2</span><span class="p">:</span><span class="n">numX</span>
    <span class="n">xx</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">xx</span><span class="p">[</span><span class="n">i</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">h</span><span class="p">;</span>
    <span class="n">yy</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">yy</span><span class="p">[</span><span class="n">i</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">h</span><span class="p">;</span>
<span class="k">end</span>
<span class="n">grid_x</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="n">xx</span><span class="p">,</span> <span class="n">j</span> <span class="k">in</span> <span class="n">yy</span><span class="p">];</span>
<span class="n">grid_y</span> <span class="o">=</span> <span class="p">[</span><span class class="n">j</span> <span class="k">for</span> <span class class="n">i</span> <span class="k">in</span> <span class class="n">xx</span><span class class="p">,</span> <span class class="n">j</span> <span class class="k">in</span> <span class class="n">yy</span><span class class="p">];</span>
<span class class="n">xy</span> <span class class="o">=</span> <span class class="p">[</span><span class class="n">grid_x</span><span class class="p">[:]</span> <span class class="n">grid_y</span><span class class="p">[:]];</span>
<span class class="n">z0</span> <span class class="o">=</span> <span class class="n">xy</span><span class class="o">*</span><span class class="n">W1</span><span class class="o">.</span><span class class="n">value</span> <span class class="o">.+</span> <span class class="n">b1</span><span class class="o">.</span><span class class="n">value</span>
<span class class="n">z0</span><span class class="p">[</span><span class class="n">z0</span> <span class class="o">.&lt;</span> <span class class="mi">0</span><span class class="p">]</span> <span class class="o">=</span> <span class class="mi">0</span> 
<span class class="n">z</span> <span class class="o">=</span> <span class class="n">z0</span><span class class="o">*</span><span class class="n">W2</span><span class class="o">.</span><span class class="n">value</span> <span class class="o">.+</span> <span class class="n">b2</span><span class class="o">.</span><span class class="n">value</span>
<span class class="n">zz</span> <span class class="o">=</span> <span class class="n">zeros</span><span class class="p">(</span><span class class="n">size</span><span class class="p">(</span><span class class="n">z</span><span class class="p">,</span><span class class="mi">1</span><span class class="p">));</span>
<span class class="k">for</span> <span class class="n">i</span> <span class class="k">in</span> <span class class="mi">1</span><span class class="p">:</span><span class class="n">size</span><span class class="p">(</span><span class class="n">z</span><span class class="p">,</span><span class class="mi">1</span><span class class="p">)</span>
    <span class class="n">zz</span><span class class="p">[</span><span class class="n">i</span><span class class="p">]</span> <span class class="o">=</span> <span class class="n">indmax</span><span class class="p">(</span><span class class="n">z</span><span class class="p">[</span><span class class="n">i</span><span class class="p">,:])</span>
<span class class="k">end</span>
<span class class="n">zz</span> <span class class="o">=</span> <span class class="n">reshape</span><span class class="p">(</span><span class class="n">zz</span><span class class="p">,</span> <span class class="n">size</span><span class class="p">(</span><span class class="n">grid_x</span><span class class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="n">contourf</span><span class="p">(</span><span class="n">grid_x</span><span class="p">,</span> <span class class="n">grid_y</span><span class class="p">,</span> <span class class="n">zz</span><span class class="p">,</span> <span class class="n">cmap</span><span class class="o">=</span><span class class="n">get_cmap</span><span class class="p">(</span><span class class="s">"Spectral"</span><span class class="p">),</span> <span class class="n">alpha</span><span class class="o">=</span><span class class="mf">0.8</span><span class class="p">)</span> 
<span class class="n">scatter</span><span class class="p">(</span><span class class="n">X</span><span class class="o">.</span><span class class="n">value</span><span class class="p">[:,</span> <span class class="mi">1</span><span class class="p">],</span> <span class class="n">X</span><span class class="o">.</span><span class class="n">value</span><span class class="p">[:,</span> <span class class="mi">2</span><span class class="p">],</span> <span class class="n">c</span><span class class="o">=</span><span class class="n">y</span><span class class="p">,</span> <span class class="n">s</span><span class class="o">=</span><span class class="mi">40</span><span class class="p">)</span>
</pre></div>

</div>

definition of the oldString included some markup pattern; we've replaced that with the intended content from the original file. Now we need to append the final paragraphs 'There is a lot missing...' etc. Let's continue reading further lines to see if they are present now.   The read continues and indeed we see 'There is a lot missing...' in original. Let's read further lines 130-220 to see.

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Moving forward on the last post, I implemented a <a href="https://github.com/lakshgupta/ToyAD.jl">toy library</a> to let us write neural networks using reverse-mode automatic differentiation. Just to show how to use the library I am using the minimal neural network example from Andrej Karpathy's <a href="http://cs231n.github.io/neural-networks-case-study/#net">CS231n class</a>. If you have already read Karpathy's notes, then the following code should be straight-forward to understand.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="k">using</span> <span class="n">ToyAD</span>
<span class="k">using</span> <span class="n">PyPlot</span>

<span class="n">N</span> <span class="o">=</span> <span class="mi">100</span> <span class="c"># number of points per class</span>
<span class="n">D</span> <span class="o">=</span> <span class="mi">2</span> <span class="c"># dimensionality</span>
<span class="n">K</span> <span class="o">=</span> <span class="mi">3</span> <span class="c"># number of classes</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="o">*</span><span class="n">K</span><span class="p">,</span> <span class="n">D</span><span class="p">))</span> <span class="c"># data matrix (each row = single example)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">N</span><span class="o">*</span><span class="n">K</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span> <span class="c"># class labels</span>
<span class="k">for</span> <span class="n">j</span> <span class="k">in</span> <span class="n">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="n">K</span><span class="p">)</span>
    <span class="n">idx</span> <span class="o">=</span> <span class="n">range</span><span class="p">(</span><span class="mi">1</span><span class="o">+</span><span class="n">N</span><span class="o">*</span><span class="p">(</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="p">),</span> <span class="n">N</span><span class="p">);</span> <span class="c">#index for X and Y</span>
    <span class="n">r</span> <span class="o">=</span> <span class="n">linspace</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="n">N</span><span class="p">);</span> <span class="c"># radius</span>
    <span class="n">t</span> <span class="o">=</span> <span class="n">linspace</span><span class="p">((</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span><span class="o">*</span><span class="mi">4</span><span class="p">,(</span><span class="n">j</span><span class="p">)</span><span class="o">*</span><span class="mi">4</span><span class="p">,</span><span class="n">N</span><span class="p">)</span> <span class="o">+</span> <span class="n">randn</span><span class="p">(</span><span class="n">N</span><span class="p">)</span><span class="o">*</span><span class="mf">0.2</span> 
    <span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[</span><span class="n">idx</span><span class="p">,</span> <span class="p">:]</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="o">.*</span><span class="n">sin</span><span class="p">(</span><span class="n">t</span><span class="p">)</span> <span class="n">r</span><span class="o">.*</span><span class="n">cos</span><span class="p">(</span><span class="n">t</span><span class="p">)]</span>
    <span class="n">y</span><span class="p">[</span><span class="n">idx</span><span class="p">,</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">j</span><span class="p">;</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># lets visualize the data:</span>
<span class="n">scatter</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">],</span>  <span class="n">s</span><span class="o">=</span><span class="mi">40</span><span class="p">,</span> <span class="n">c</span><span class="o">=</span><span class="n">y</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAArYAAAIUCAYAAADv128JAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIABJREFUeJzs3Xl0VXWe7/33GZKTOWQmIxkghCEkZGSSQVAQnMqZciiHaq3usu22+nbdatfT7epL913r1u0qn6e6hta2REEFS8BSUQFBQKYQMkAgJISQeQ6Z5+Gc8/yBRlMEBczE8fNayz88++y9vycnnHzOb//292ew2+12RERERERucMaJLkBEREREZDQo2IqIiIiIQ1CwFRERERGHoGArIiIiIg5BwVZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDGPNg29XVxYsvvsiaNWvw9fXFaDTyxhtvXNW+r7/+OkajccT/GhoaxrhyEREREbmRmMf6BI2NjWzYsIFp06aRmJjIgQMHMBgM13SMDRs2EBUVNewxb2/v0SxTRERERG5wYx5sQ0JCqKurIzAwkOzsbFJTU6/5GLfddhtJSUljUJ2IiIiIOIoxn4rg7OxMYGAgAHa7/bqOYbfb6ejowGq1jmZpIiIiIuJAboibx1asWIG3tzfu7u7cddddFBcXT3RJIiIiIjLJjPlUhO/C3d2dJ554ghUrVuDl5UVWVha//vWvWbRoETk5OYSFhU10iSIiIiIySUzqYHv//fdz//33D/3/nXfeyerVq1m6dCn//u//zh/+8IfL9rl48SK7d+8mMjISV1fX8SxXRERERK5CT08PZWVlrF69Gn9//1E77qQOtiNZvHgx6enp7N27d8Ttu3fv5pFHHhnnqkRERETkWr355ps8/PDDo3a8Gy7YAoSFhVFUVDTitsjISODSD2rWrFnjWJVMlOeff56XXnpposuQcaL3+/tF7/f3i97v74+CggIeeeSRodw2Wm7IYFtSUkJAQMCI276cfjBr1iy1CPue8Pb21nv9PaL3+/tF7/f3i97v75/RnjY6aboi1NXVUVhYyODg4NBjjY2Nlz3v448/JicnhzVr1oxneSIiIiIyyY3LiO1vf/tbWltbqampAeCDDz6goqICgOeeew4vLy9+8YtfsGnTJsrKyoiIiABg0aJFJCUlkZycjLe3Nzk5Obz22mtERETwwgsvjEfpIiIiInKDGJdg+6tf/Yry8nIADAYD7733Hjt27MBgMPDYY4/h5eWFwWC4bKndhx56iI8++og9e/bQ3d1NSEgIzzzzDC+++OIVpyKIiIiIyPfTuATb0tLSb33Oxo0b2bhx47DHNmzYwIYNG8aqLHEQ69evn+gSZBzp/f5+0fv9/aL3W76rSTPHVuR66YPw+0Xv9/eL3u/vF73f8l0p2IqIiIiIQ1CwFRERERGHoGArIiIiIg5BwVZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiEMY82Db1dXFiy++yJo1a/D19cVoNPLGG29c9f6tra08/fTTBAQE4OHhwc0330xubu4YViwiIiIiN6IxD7aNjY1s2LCBc+fOkZiYCIDBYLiqfW02G+vWrWPLli0899xz/PKXv6ShoYHly5dTXFw8lmWLiIiIyA3GPNYnCAkJoa6ujsDAQLKzs0lNTb3qfbdt28axY8fYtm0b99xzDwAPPPAAsbGxvPjii7z11ltjVbaIiIiI3GDGfMTW2dmZwMBAAOx2+zXtu23bNqZOnToUagH8/f154IEHeP/99xkYGBjVWkVERETkxjWpbx7Lzc0lKSnpssdTU1Pp7u6mqKhoAqoScUxWq5X8/Hz+/Oc/s337dg4dOkRnZ+dElyUiInLVxnwqwndRW1vL8uXLL3s8ODgYgJqaGubMmTPOVYk4noaGBt744x+pOH0al95enIxG2g0GPg4N5Qc//CFpaWkTXaKIiMi3mtTBtre3F4vFctnjLi4uAPT09Ix3SSIOp7u7m1d//3uaT51iRUwMvh4eAPQNDHCyrIytr7yCu7u7vkSKiMikN6mDraurK319fZc93tvbO7T9Sp5//nm8vb2HPbZ+/XrWr18/ukWK3OCys7OpOXOGtXFxuH3ti6TFyYm06dP57OxZ9u3Zw+zZs6+6o4mIiMiXtmzZwpYtW4Y91tbWNibnmtTBNjg4mJqamsser62tBS51XLiSl156acT5uSIy3MkTJwgym4eF2i8ZDAbigoPJOnOGhoYGgoKCJqBCERG5kY00sJiTk0NycvKon2tS3zyWmJhITk7OZd0Ujh8/jru7O7GxsRNUmYjj6Ghrw/OL6T0j8XBxwdrXp6k/IiIy6U2aYFtXV0dhYSGDg4NDj913333U19ezY8eOoccuXrzIu+++yx133IGTk9NElCriUHwDA2nu6rri9ubOTpzc3PDy8hrHqkRERK7duExF+O1vf0tra+vQtIIPPviAiooKAJ577jm8vLz4xS9+waZNmygrKyMiIgK4FGwXLFjAE088wdmzZ/Hz8+P3v/89drudf/3Xfx2P0kUcXkp6Oq9//jktXV34uLsP22a12SioqyNu9Wp8fX0nqEIREZGrMy7B9le/+hXl5eXApTl77733Hjt27MBgMPDYY4/h5eWFwWC47MYUo9HIxx9/zD/+4z/ym9/8hp6eHtLS0ti0aRMzZswYj9JFHN68efOYuXAh+w8eZH5wMBH+/piMRpo6OsgtL8cUHs6ta9ZMdJkiIiLfymC/1uXAJrkvJyNnZ2fr5jGRq9TV1cW2P/2JU0ePMtDSgslgwGaxEDJzJvetX8/06dMnukQREXEgY5XXJnVXBBEZH+7u7vzoiSdoWLeO8+fPY7VaCQwMJDY2FqNx0kzFFxER+UYKtiIyJDAwkMDAwIkuQ0RE5LpoKEZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiEMY82Db1dXFiy++yJo1a/D19cVoNPLGG29c9f6tra08/fTTBAQE4OHhwc0330xubu4YViwiIiIiN6IxD7aNjY1s2LCBc+fOkZiYCIDBYLiqfW02G+vWrWPLli0899xz/PKXv6ShoYHly5dTXFw8lmWLiIiIyA3GPNYnCAkJoa6ujsDAQLKzs0lNTb3qfbdt28axY8fYtm0b99xzDwAPPPAAsbGxvPjii7z11ltjVbaIiIiI3GDGfMTW2dmZwMBAAOx2+zXtu23bNqZOnToUagH8/f154IEHeP/99xkYGBjVWkVERETkxjWpbx7Lzc0lKSnpssdTU1Pp7u6mqKhoAqoScUxWq5X8/Hz+/Oc/s337dg4dOkRnZ+dElyUiInLVxnwqwndRW1vL8uXLL3s8ODgYgJqaGubMmTPOVYk4noaGBt744x+pOH0al95enIxG2g0GPg4N5Qc//CFpaWkTXaKIiMi3mtTBtre3F4vFctnjLi4uAPT09Ix3SSIOp7u7m1d//3uaT51iRUwMvh4eAPQNDHCyrIytr7yCu7u7vkSKiMikN6mDraurK319fZc93tvbO7T9Sp5//nm8vb2HPbZ+/XrWr18/ukWK3OCys7OpOXOGtXFxuH3ti6TFyYm06dP57OxZ9u3Zw+zZs6+6o4mIiMiXtmzZwpYtW4Y91tbWNibnmtTBNjg4mJqamsser62tBS51XLiSl156acT5uSIy3MkTJwgym4eF2i8ZDAbigoPJOnOGhoYGgoKCJqBCERG5kY00sJiTk0NycvKon2tS3zyWmJhITk7OZd0Ujh8/jru7O7GxsRNUmYjj6Ghrw/OL6T0j8XBxwdrXp6k/IiIy6U2aYFtXV0dhYSGDg4NDj913333U19ezY8eOoccuXrzIu+++yx133IGTk9NElCriUHwDA2nu6rri9ubOTpzc3PDy8hrHqkRERK7duExF+O1vf0tra+vQtIIPPviAiooKAJ577jm8vLz4xS9+waZNmygrKyMiIgK4FGwXLFjAE088wdmzZ/Hz8+P3v/89drudf/3Xfx2P0kUcXkp6Oq9//jktXV34uLsP22a12SioqyNu9Wp8fX0nqEIREZGrMy7B9le/+hXl5eXApTl77733Hjt27MBgMPDYY4/h5eWFwWC47MYUo9HIxx9/zD/+4z/ym9/8hp6eHtLS0ti0aRMzZswYj9JFHN68efOYuXAh+w8eZH5wMBH+/piMRpo6OsgtL8cUHs6ta9ZMdJkiIiLfymC/1uXAJrkvJyNnZ2fr5jGRq9TV1cW2P/2JU0ePMtDSgslgwGaxEDJzJvetX8/06dMnukQREXEgY5XXJnVXBBEZH+7u7vzoiSdoWLeO8+fPY7VaCQwMJDY2FqNx0kzFFxER+UYKtiIyJDAwkMDAwIkuQ0RE5LpoKEZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiEMY82Db1dXFiy++yJo1a/D19cVoNPLGG29c9f6tra08/fTTBAQE4OHhwc0330xubu4YViwiIiIiN6IxD7aNjY1s2LCBc+fOkZiYCIDBYLiqfW02G+vWrWPLli0899xz/PKXv6ShoYHly5dTXFw8lmWLiIiIyA3GPNYnCAkJoa6ujsDAQLKzs0lNTb3qfbdt28axY8fYtm0b99xzDwAPPPAAsbGxvPjii7z11ltjVbaIiIiI3GDGfMTW2dmZwMBAAOx2+zXtu23bNqZOnToUagH8/f154IEHeP/99xkYGBjVWkVERETkxjWpbx7Lzc0lKSnpssdTU1Pp7u6mqKhoAqoScUxWq5X8/Hz+/Oc/s337dg4dOkRnZ+dElyUiInLVxnwqwndRW1vL8uXLL3s8ODgYgJqaGubMmTPOVYk4noaGBt744x+pOH0al95enIxG2g0GPg4N5Qc//CFpaWkTXaKIiMi3mtTBtre3F4vFctnjLi4uAPT09Ix3SSIOp7u7m1d//3uaT51iRUwMvh4eAPQNDHCyrIytr7yCu7u7vkSKiMikN6mDraurK319fZc93tvbO7T9Sp5//nm8vb2HPbZ+/XrWr18/ukWK3OCys7OpOXOGtXFxuH3ti6TFyYm06dP57OxZ9u3Zw+zZs6+6o4mIiMiXtmzZwpYtW4Y91tbWNibnmtTBNjg4mJqamsser62tBS51XLiSl156acT5uSIy3MkTJwgym4eF2i8ZDAbigoPJOnOGhoYGgoKCJqBCERG5kY00sJiTk0NycvKon2tS3zyWmJhITk7OZd0Ujh8/jru7O7GxsRNUmYjj6Ghrw/OL6T0j8XBxwdrXp6k/IiIy6U2aYFtXV0dhYSGDg4NDj913333U19ezY8eOoccuXrzIu+++yx133IGTk9NElCriUHwDA2nu6rri9ubOTpzc3PDy8hrHqkRERK7duExF+O1vf0tra+vQtIIPPviAiooKAJ577jm8vLz4xS9+waZNmygrKyMiIgK4FGwXLFjAE088wdmzZ/Hz8+P3v/89drudf/3Xfx2P0kUcXkp6Oq9//jktXV34uLsP22a12SioqyNu9Wp8fX0nqEIREZGrMy7B9le/+hXl5eXApTl77733Hjt27MBgMPDYY4/h5eWFwWC47MYUo9HIxx9/zD/+4z/ym9/8hp6eHtLS0ti0aRMzZswYj9JFHN68efOYuXAh+w8eZH5wMBH+/piMRpo6OsgtL8cUHs6ta9ZMdJkiIiLfymC/1uXAJrkvJyNnZ2fr5jGRq9TV1cW2P/2JU0ePMtDSgslgwGaxEDJzJvetX8/06dMnukQREXEgY5XXJnVXBBEZH+7u7vzoiSdoWLeO8+fPY7VaCQwMJDY2FqNx0kzFFxER+UYKtiIyJDAwkMDAwIkuQ0RE5LpoKEZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiEMY82Db1dXFiy++yJo1a/D19cVoNPLGG29c9f6tra08/fTTBAQE4OHhwc0330xubu4YViwiIiIiN6IxD7aNjY1s2LCBc+fOkZiYCIDBYLiqfW02G+vWrWPLli0899xz/PKXv6ShoYHly5dTXFw8lmWLiIiIyA3GPNYnCAkJoa6ujsDAQLKzs0lNTb3qfbdt28axY8fYtm0b99xzDwAPPPAAsbGxvPjii7z11ltjVbaIiIiI3GDGfMTW2dmZwMBAAOx2+zXtu23bNqZOnToUagH8/f154IEHeP/99xkYGBjVWkVERETkxjWpbx7Lzc0lKSnpssdTU1Pp7u6mqKhoAqoScUxWq5X8/Hz+/Oc/s337dg4dOkRnZ+dElyUiInLVxnwqwndRW1vL8uXLL3s8ODgYgJqaGubMmTPOVYk4noaGBt744x+pOH0al95enIxG2g0GPg4N5Qc//CFpaWkTXaKIiMi3mtTBtre3F4vFctnjLi4uAPT09Ix3SSIOp7u7m1d//3uaT51iRUwMvh4eAPQNDHCyrIytr7yCu7u7vkSKiMikN6mDraurK319fZc93tvbO7T9Sp5//nm8vb2HPbZ+/XrWr18/ukWK3OCys7OpOXOGtXFxuH3ti6TFyYm06dP57OxZ9u3Zw+zZs6+6o4mIiMiXtmzZwpYtW4Y91tbWNibnmtTBNjg4mJqamsser62tBS51XLiSl156acT5uSIy3MkTJwgym4eF2i8ZDAbigoPJOnOGhoYGgoKCJqBCERG5kY00sJiTk0NycvKon2tS3zyWmJhITk7OZd0Ujh8/jru7O7GxsRNUmYjj6Ghrw/OL6T0j8XBxwdrXp6k/IiIy6U2aYFtXV0dhYSGDg4NDj913333U19ezY8eOoccuXrzIu+++yx133IGTk9NElCriUHwDA2nu6rri9ubOTpzc3PDy8hrHqkRERK7duExF+O1vf0tra+vQtIIPPviAiooKAJ577jm8vLz4xS9+waZNmygrKyMiIgK4FGwXLFjAE088wdmzZ/Hz8+P3v/89drudf/3Xfx2P0kUcXkp6Oq9//jktXV34uLsP22a12SioqyNu9Wp8fX0nqEIREZGrMy7B9le/+hXl5eXApTl77733Hjt27MBgMPDYY4/h5eWFwWC47MYUo9HIxx9/zD/+4z/ym9/8hp6eHtLS0ti0aRMzZswYj9JFHN68efOYuXAh+w8eZH5wMBH+/piMRpo6OsgtL8cUHs6ta9ZMdJkiIiLfymC/1uXAJrkvJyNnZ2fr5jGRq9TV1cW2P/2JU0ePMtDSgslgwGaxEDJzJvetX8/06dMnukQREXEgY5XXJnVXBBEZH+7u7vzoiSdoWLeO8+fPY7VaCQwMJDY2FqNx0kzFFxER+UYKtiIyJDAwkMDAwIkuQ0RE5LpoKEZEREREHIKCrYiIiIg4BAVbEREREXEICrYiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiENQsBURERERh6BgKyIiIiIOQcFWRERERByCgq2IiIiIOAQFWxERERFxCAq2IiIiIi4hAUbEVERETEISjYioiIiIhDULAVEREREYegYCsiIiIiDkHBVkREREQcgoKtiIiIiDgEBVsRERERcQgKtiIiIiLiEBRsRURERMQhKNiKiIiIiEMY82Db1dXFiy++yJo1a/D19cVo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">
</div>

</div>
<div class="output_area"><div class="prompt output_prompt">Out[3]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>PyObject &lt;matplotlib.collections.PathCollection object at 0x000000002B6800B8&gt;</pre>
</div>

</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We operate on the same type AD as in the last post. We build the computational graph and use the chain rule applying the reverse-mode automatic differentiation to calculate the gradients. The name of the function which does this is 'backprop'. There are few other functions such as relu and softmaxLoss, which I implemented in the library as and when required for finishing this example.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="k"># initialize the parameters</span>
<span class="n">W1</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">randn</span><span class="p">(</span><span class="n">D</span><span class="o">,</span><span class="mi">100</span><span class="p">))</span>
<span class="n">W2</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">randn</span><span class="p">(</span><span class="mi">100</span><span class="o">,</span><span class="n">K</span><span class="p">))</span>
<span class="n">b1</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="mi">1</span><span class="o">,</span><span class="mi">100</span><span class="p">))</span>
<span class="n">b2</span> <span class="o">=</span> <span class="n">AD</span><span class="p">(</span><span class="n">zeros</span><span class="p">(</span><span class="mi">1</span><span class="o">,</span><span class="n">K</span><span class="p">))</span>

<span class="n">lr</span> <span class="o">=</span> <span class="mf">1e-0</span>
<span class="n">reg</span> <span class="o">=</span> <span class="mf">1e-3</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">20000</span>

<span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">epochs</span>
    <span class="n">z0</span> <span class="o">=</span> <span class="n">X</span><span class="o">*</span><span class="n">W1</span><span class="o">.</span><span class="n">value</span> <span class="o">.+</span> <span class="n">b1</span><span class="o">.</span><span class="n">value</span>
    <span class="n">z0</span><span class="p">[</span><span class="n">z0</span> <span class="o">.&lt;</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">z</span> <span class="o">=</span> <span class="n">z0</span><span class="o">*</span><span class="n">W2</span><span class="o">.</span><span class="n">value</span> <span class="o">.+</span> <span class="n">b2</span><span class="o">.</span><span class="n">value</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="n">softmaxLoss</span><span class="p">(</span><span class="n">z</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span> <span class="o">+</span> <span class="n">reg</span><span class="o">.*</span><span class="n">sum</span><span class="p">(</span><span class="n">W1</span><span class="o">.^</span><span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="n">reg</span><span class="o">.*</span><span class="n">sum</span><span class="p">(</span><span class="n">W2</span><span class="o">.^</span><span class="mi">2</span><span class="p">)</span>

    <span class="n">grads</span> <span class="o">=</span> <span class="n">backprop</span><span class="p">(</span><span class="n">loss</span><span class="p">)</span>

    <span class="n">W1</span><span class="o">.</span><span class="n">value</span> <span class="o">-=</span> <span class="n">lr</span><span class="o">.*</span><span class="n">grads</span><span class="p">[</span><span class="s">"W1"</span><span class="p">]</span>
    <span class="n">W2</span><span class="o">.</span><span class="n">value</span> <span class="o">-=</span> <span class="n">lr</span><span class="o">.*</span><span class="n">grads</span><span class="p">[</span><span class="s">"W2"</span><span class="p">]</span>
    <span class="n">b1</span><span class="o">.</span><span class="n">value</span> <span class="o">-=</span> <span class="n">lr</span><span class="o">.*</span><span class="n">grads</span><span class="p">[</span><span class="s">"b1"</span><span class="p">]</span>
    <span class="n">b2</span><span class="o">.</span><span class="n">value</span> <span class="o">-=</span> <span class="n">lr</span><span class="o">.*</span><span class="n">grads</span><span class="p">[</span><span class="s">"b2"</span><span class="p">]</span>

    <span class="k">if</span> <span class="n">mod</span><span class="p">(</span><span class="n">i</span><span class="p">,</span> <span class="mi">500</span><span class="p">)</span> <span class="o">==</span> <span class="mi">0</span>
        <span class="n">println</span><span class="p">(</span><span class="s">"iter: "</span> <span class="o">*</span> <span class="n">string</span><span class="p">(</span><span class="n">i</span><span class="p">)</span> <span class="o">*</span> <span class="s">" loss: "</span> <span class="o">*</span> <span class="n">string</span><span class="p">(</span><span class="n">loss</span><span class="p">)</span> <span class="p">)</span>
    <span class="k">end</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># evaluate on a grid</span>
<span class="n">h</span> <span class="o">=</span> <span class="mf">0.02</span><span class="p">;</span>
<span class="n">x_min</span> <span class="o">=</span> <span class="n">minimum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">])</span> <span class="o">-</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">x_max</span> <span class="o">=</span> <span class="n">maximum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">])</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">y_min</span> <span class="o">=</span> <span class="n">minimum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">])</span> <span class="o">-</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">y_max</span> <span class="o">=</span> <span class="n">maximum</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">])</span> <span class="o">+</span> <span class="mi">1</span><span class="p">;</span>
<span class="n">numX</span> <span class="o">=</span> <span class="nb">convert</span><span class="p">(</span><span class="kt">Int</span><span class="p">,</span> <span class="n">floor</span><span class="p">((</span><span class="n">x_max</span> <span class="o">-</span> <span class="n">x_min</span><span class="p">)</span><span class="o">/</span><span class="n">h</span><span class="p">));</span>
<span class="n">xx</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">numX</span><span class="p">);</span>
<span class="n">xx</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">x_min</span><span class="p">;</span>
<span class="n">yy</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">numX</span><span class="p">);</span>
<span class="n">yy</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">y_min</span><span class="p">;</span>
<span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">2</span><span class="p">:</span><span class="n">numX</span>
    <span class="n">xx</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">xx</span><span class="p">[</span><span class="n">i</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">h</span><span class="p">;</span>
    <span class="n">yy</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">yy</span><span class="p">[</span><span class="n">i</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">h</span><span class="p">;</span>
<span class="k">end</span>
<span class="n">grid_x</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="n">xx</span><span class="p">,</span> <span class="n">j</span> <span class="k">in</span> <span class="n">yy</span><span class="p">];</span>
<span class="n">grid_y</span> <span class="o">=</span> <span class="p">[</span><span class="n">j</span> <span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="n">xx</span><span class="p">,</span> <span class="n">j</span> <span class="k">in</span> <span class="n">yy</span><span class="p">];</span>
<span class="n">xy</span> <span class="o">=</span> <span class="p">[</span><span class="n">grid_x</span><span class="p">[:]</span> <span class="n">grid_y</span><span class="p">[:]];</span>
<span class="n">z0</span> <span class="o">=</span> <span class="n">xy</span><span class="o">*</span><span class="n">W1</span><span class="o">.</span><span class="n">value</span> <span class="o">.+</span> <span class="n">b1</span><span class="o">.</span><span class="n">value</span>
<span class="n">z0</span><span class="p">[</span><span class="n">z0</span> <span class="o">.&lt;</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span> 
<span class="n">z</span> <span class="o">=</span> <span class="n">z0</span><span class="o">*</span><span class="n">W2</span><span class="o">.</span><span class="n">value</span> <span class="o">.+</span> <span class="n">b2</span><span class="o">.</span><span class="n">value</span>
<span class="n">zz</span> <span class="o">=</span> <span class="n">zeros</span><span class="p">(</span><span class="n">size</span><span class="p">(</span><span class="n">z</span><span class="p">,</span><span class="mi">1</span><span class="p">));</span>
<span class="k">for</span> <span class="n">i</span> <span class="k">in</span> <span class="mi">1</span><span class="p">:</span><span class="n">size</span><span class="p">(</span><span class="n">z</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">zz</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">indmax</span><span class="p">(</span><span class="n">z</span><span class="p">[</span><span class="n">i</span><span class="p">,:])</span>
<span class="k">end</span>
<span class="n">zz</span> <span class="o">=</span> <span class="n">reshape</span><span class="p">(</span><span class="n">zz</span><span class="p">,</span> <span class="n">size</span><span class="p">(</span><span class="n">grid_x</span><span class="p">));</span>
</pre></div>

</div>
</div>
</div>

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="n">contourf</span><span class="p">(</span><span class="n">grid_x</span><span class="p">,</span> <span class="n">grid_y</span><span class="p">,</span> <span class="n">zz</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">get_cmap</span><span class="p">(</span><span class="s">"Spectral"</span><span class="p">),</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.8</span><span class="p">)</span> 
<span class="n">scatter</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">X</span><span class="o">.</span><span class="n">value</span><span class="p">[:,</span> <span class="mi">2</span><span class="p">],</span> <span class="n">c</span><span class="o">=</span><span class="n">y</span><span class="p">,</span> <span class="n">s</span><span class="o">=</span><span class="mi">40</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApoAAAILCAYAAABINjwZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIABJREFUeJzs3Xdc1WX/x/HXOewNgoCICu49cK/Ebe7MnaO9l3fz1zJv7/Zd1p1NM3dmmXvv3BP3XrgQEZW9Od/fHydRAsqKA4jv5+PhI/vO64tHfHNd3+tzmQzDMBARERERKWTm4m6AiIiIiJROCpoiIiIiYhMKmiIiIiJiEwqaIiIiImITCpoiIiIiYhMKmiIiIiJiEwqaIiIiImIT9sXdgN+LjY1l+fLlhISE4OLiUtzNEREREZHfSU1NJTIykq5du+Ln51fgcSUuaC5fvpxhw4YVdzNERERE5E9Mnz6d++67r8D9JS5ohoSEANaG16pVq8DjRo0axbhx44qoVVKa6bMkhUWfJSkM+hxJYbHlZ+nw4cMMGzYsJ7cVpMQFzevD5bVq1SIsLKzA47y8vP5wv8it0mdJCos+S1IY9DmSwlIUn6U/e81Rk4FERERExCYUNEVERETEJhQ0RURERMQmbtugOWTIkOJugpQS+ixJYdFnSQqDPkdSWErCZ8lkGIZR3I24WUREBI0bN2bXrl16GVpERESkBLrVvGbzHs0dO3bw9NNPU6dOHdzd3alUqRKDBg3i+PHjtr61iIiIiBQjm5c3+uCDD9iyZQsDBgygfv36XLx4kfHjxxMWFsbWrVupU6eOrZsgIiIiIsXA5kHzhRdeoGnTptjb37jVoEGDqFevHu+//z7Tpk2zdRNEREREpBjYPGi2bNkyz7aqVatSu3Ztjhw5Yuvbi4iIiEgxKZZZ54ZhcOnSpT9chF1EREREbm/FEjRnzJhBVFQUgwYNKo7bi4iIiEgRKPKgeeTIEZ566ilatWrFyJEji/r2IiIiIlJEijRoRkdH06NHD3x8fJg9ezYmk6koby8iIiIiRcjmk4Gui4+P5+677yYhIYENGzYQGBj4h8ePGjUKLy+vXNuGDBlSIqrci4iIiNwpZs6cycyZM3Nti4+Pv6Vzi2RloLS0NLp06cLu3btZtWoVzZs3L/BYrQwkIiIiUrLdal6zeY9mdnY2gwYNYtu2bcyfP/8PQ6aIiIiIlB5FUrB94cKF9OrVi9jYWKZPn55r/7Bhw2zdBBEREREpBjYPmnv37sVkMrFw4UIWLlyYa5/JZFLQFBERESmlbB40165da+tbiIiIiEgJVGSzzkuqJz9ZX9xNEJFCFJWcQOV2yYxpCu7prsXdHBGRImPy7lXcTcijWFYGEhEREZHST0FTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJBU0RERERsQkFTRERERGxCQVNEREREbEJmwfN5ORkRo8eTbdu3ShTpgxms5kpU6bY+rYiIiIiUsxsHjQvX77M2LFjOXr0KA0bNgTAZDLZ+rYiIiIiUszsbX2DoKAgoqOj8ff3Z9euXTRt2tTWtxQRERGREsDmQdPR0RF/f38ADMOw9e1ERPIwLBYunNhNwtUoHJxcsQ+uDmitcxERW7N50BQRKU6Rhzaz6ZdxJMTF5Gyzs3fA42oXvv5oSDG2TESk9FPQFJFS68yhLSz7/jWqGNAfCAaSgF1ZmfwyYQmm+DhmfvWy3hsXEbERlTcSkVLJsFjYOGccVQwYikFFrN/wPIH2QB/D4KdZm9i642jxNlREpBQrsT2ao0aNwsvLK9e2IUOGMGSIhrpE5M9FndpDwrVL9Cf/n6jrAevtzUyYspyWzWoWcetERG4fM2fOZObMmbm2xcfH39K5JTZojhs3jrCwsOJuhojcpuJjowDrcHl+zEBQloWTp6KKrE0iIrej/Dr6IiIiaNy48Z+eq6FzESmVHJ1dAes7mQVJMpvw9HQrm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...