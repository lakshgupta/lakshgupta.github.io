---
title: "Automatic Differentiation using Operator Overloading"
subtitle: "Automatic differentiation, derivatives"
description: "Implementing automatic differentiation using operator overloading and forward/reverse modes."
date: 2016-11-06T12:00:00
author: "Laksh Gupta"
tags: ["automatic-differentiation", "machine-learning"]
---
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Almost all of the libraries for creating neural networks (Tensorflow, Theano, Torch, etc.) are using automatic differentiation (AD) in one way or another. It has applications in the other parts of the mathematical world as well since it is a clever and effective way to calculate the gradients, effortlessly. It works by first creating a computational graph of the operations and then traversing it in either forward mode or reverse mode. Let us see how to implement them using operator overloading to calculate the first order partial derivative. I highly recommend reading Colah's blog <a href="http://colah.github.io/posts/2015-08-Backprop/">here</a> first. It has an excellent explanation about computational graphs and this post is related to the implementation side of it. It may not be the best performing piece of code for AD but I think it's the simplest one for getting your head around the concept. The example function we are considering here is:</p>
$$f(a, b) = (a + b) * (b + 1)$$<p>Using basic differential calculus rules, we can calculate the derivative of the above function with respect to a and b by applying the <a href="https://en.wikipedia.org/wiki/Sum_rule_in_differentiation">sum rule</a> and the <a href="https://en.wikipedia.org/wiki/Product_rule">product rule</a>:</p>
$$\begin{eqnarray}
\frac{\partial e}{\partial a}
&=&(c*\frac{\partial d}{\partial a}) + (\frac{\partial c}{\partial a}*d) \\ \nonumber
&=&((a + b)* (\frac{\partial b}{\partial a} + \frac{\partial 1}{\partial a}) )+ ((\frac{\partial a}{\partial a} + \frac{\partial b}{\partial a})* (b + 1)) \\ \nonumber 
&=&(1 + 0)* (b + 1)) \\ \nonumber 
&=&b + 1 \\ \nonumber 
\end{eqnarray}
$$<p>and similarly,</p>
$$\begin{eqnarray}
\frac{\partial e}{\partial b}
&=&(c*\frac{\partial d}{\partial b}) + (\frac{\partial c}{\partial b}*d) \\ \nonumber
&=&((a + b)* (\frac{\partial b}{\partial b} + \frac{\partial 1}{\partial b})) + ((\frac{\partial a}{\partial b} + \frac{\partial b}{\partial b})* (b + 1)) \\ \nonumber 
&=&((a + b)*(1 + 0)) * ((0 + 1)*(b + 1)) \\ \nonumber 
&=&(a + b)*(b + 1) \\ \nonumber 
\end{eqnarray}
$$<p>In order to get the derivative of a function programmatically there are two approaches we can follow and operate on the computational graph, forward mode and reverse mode. Both of these approaches make use of the <a href="https://en.wikipedia.org/wiki/Chain_rule">chain rule</a>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># for overloading</span>
<span class="k">import</span> <span class="n">Base</span><span class="o">.+</span><span class="p">,</span> <span class="n">Base</span><span class="o">.*</span>
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
<h2 class="section-heading">Forward Mode</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Forward mode is very similar to the calculation we did above. We pick an independent variable with respect to which we would like to calculate the partial derivative of the function, set its derivative with respect to itself as 1 and then, we recursively moving forward calculate the derivative of the sub-graph till we reach the output node.</p>
<blockquote><p>In a pen-and-paper calculation, one can do so by repeatedly substituting the derivative of the inner functions in the chain rule:</p>
$${\displaystyle {\frac {\partial y}{\partial x}}={\frac {\partial y}{\partial w_{1}}}{\frac {\partial w_{1}}{\partial x}}={\frac {\partial y}{\partial w_{1}}}\left({\frac {\partial w_{1}}{\partial w_{2}}}{\frac {\partial w_{2}}{\partial x}}\right)={\frac {\partial y}{\partial w_{1}}}\left({\frac {\partial w_{1}}{\partial w_{2}}}\left({\frac {\partial w_{2}}{\partial w_{3}}}{\frac {\partial w_{3}}{\partial x}}\right)\right)=\cdots }$$<p></p>
<p>- <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">wikipedia</a></p>
</blockquote>
<p>The graph here can be thought to be constructed by way a programming language may perform the operations, using the <a href="https://en.wikipedia.org/wiki/Order_of_operations">BODMAS</a> rule. In terms of simple operations, the above function can be broken down to:</p>
$$c = a + b$$$$d = b + 1$$$$e = c * d$$<p>hence the operations to calculate the partial derivative of the above function with respect to a may look like:</p>
\begin{array}{cc|lcr|lcr}
\mathrm{value} && \mathrm{derivative} && node\\
\hline \\
a=a   && {\frac{\partial a}{\partial a}} = 1  && node 1\\
b=b   && {\frac{\partial b}{\partial a}} = 0  && node 2\\
c=a+b && {\frac{\partial c}{\partial a}} = {\frac{\partial a}{\partial a}} + {\frac{\partial b}{\partial a}}  && node3 \Leftarrow node1 + node2 \\\nd=b+1 && {\frac{\partial d}{\partial a}} = {\frac{\partial b}{\partial a}} + {\frac{\partial 1}{\partial a}} && node5 \Leftarrow node2 + node4  \\\ne=c*d && {\frac{\partial e}{\partial a}} = c*{\frac{\partial d}{\partial a}} + {\frac{\partial c}{\partial a}}*d && node6 \Leftarrow node3*node5 \\\n\end{array}<p>To simulate the above steps, we have a type ADFwd which also represents a node in the calculation graph.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># type to store the float value for a variable and</span>
<span class="c"># the derivative with repect to the variable at that value.</span>
<span class="k">type</span><span class="nc"> ADFwd</span>
    <span class="n">value</span><span class="p">::</span><span class="kt">Float64</span> <span class="c"># say, to store c</span>
    <span class="n">derivative</span><span class="p">::</span><span class="kt">Float64</span> <span class="c"># say, to store dc/da</span>
    
    <span class="n">ADFwd</span><span class="p">(</span><span class="n">val</span><span class="p">::</span><span class="kt">Float64</span><span class="p">)</span> <span class="o">=</span> <span class="nb">new</span><span class="p">(</span><span class="n">val</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
    <span class="n">ADFwd</span><span class="p">(</span><span class="n">val</span><span class="p">::</span><span class="kt">Float64</span><span class="p">,</span> <span class="n">der</span><span class="p">::</span><span class="kt">Float64</span><span class="p">)</span> <span class="o">=</span> <span class="nb">new</span><span class="p">(</span><span class="n">val</span><span class="p">,</span> <span class="n">der</span><span class="p">)</span>
<span class="k">end</span>
</pre></div>


</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We define the operation on this type, and also the derivation rule to follow. Operator overloading helps here in the operations over the type ADFwd.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># sum rule</span>
<span class="k">function</span><span class="nf"> adf_add</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">ADFwd</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">value</span> <span class="o">+</span> <span class="n">y</span><span class="o">.</span><span class="n">value</span><span class="p">,</span> <span class="n">x</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="n">y</span><span class="o">.</span><span class="n">derivative</span><span class="p">)</span>
<span class="k">end</span>
<span class="o">+</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">)</span> <span class="o">=</span> <span class="n">adf_add</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>

<span class="c"># product rule</span>
<span class="k">function</span><span class="nf"> adf_mul</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">ADFwd</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">value</span> <span class="o">*</span> <span class="n">y</span><span class="o">.</span><span class="n">value</span><span class="p">,</span> <span class="n">y</span><span class="o">.</span><span class="n">value</span> <span class="o">*</span> <span class="n">x</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="n">x</span><span class="o">.</span><span class="n">value</span> <span class="o">*</span> <span class="n">y</span><span class="o">.</span><span class="n">derivative</span><span class="p">)</span>
<span class="k">end</span>
<span class="o">*</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">)</span> <span class="o">=</span> <span class="n">adf_mul</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[3]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>* (generic function with 150 methods)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># define test function</span>
<span class="k">function</span><span class="nf"> f</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">,</span><span class="n">y</span><span class="p">::</span><span class="n">ADFwd</span><span class="p">)</span>
    <span class="p">(</span><span class="n">x</span><span class="o">+</span><span class="n">y</span><span class="p">)</span><span class="o">*</span><span class="p">(</span><span class="n">y</span> <span class="o">+</span> <span class="n">ADFwd</span><span class="p">(</span><span class="mf">1.0</span><span class="p">))</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[4]:</div>



</div>

</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[4]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>f (generic function with 1 method)</pre>
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
<p>Now let us get the partial derivative of the above function with respect to 'a'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># define variables</span>
<span class="n">aFwd</span> <span class="o">=</span> <span class="n">ADFwd</span><span class="p">(</span><span class="mf">2.0</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">)</span>
<span class="n">bFwd</span> <span class="o">=</span> <span class="n">ADFwd</span><span class="p">(</span><span class="mf">1.0</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[5]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>ADFwd(1.0,0.0)</pre>
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
<div class=" highlight hl-julia"><pre><span></span><span class="c"># forward mode AD</span>
<span class="n">eaFwd</span> <span class="o">=</span> <span class="n">f</span><span class="p">(</span><span class="n">aFwd</span><span class="p">,</span> <span class="n">bFwd</span><span class="p">)</span>
<span class="n">eaFwd</span><span class="o">.</span><span class="n">value</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>9.0
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># corresponding derivative</span>
<span class="n">eaFwd</span><span class="o">.</span><span class="n">derivative</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>2.0
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
<h2 class="section-heading">Reverse Mode (or the backpropagation)</h2><p>The reverse mode on the computational graph is more convenient when one wants to calculate gradient of single output function wrt many input variables. It is the algorithm used to calculate gradient for neural networks. We use the same graph here but we backpropagate(although not the same method) on the graph to update the derivatives for each node. For reverse mode implementation, I've used a slightly more complext generic type ADRev which stores the node and the pointers to the current node.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="c"># generic AD variable</span>
<span class="k">type</span><span class="nc"> ADRev</span>
    <span class="n">val</span><span class="p">::</span><span class="kt">Float64</span>
    <span class="n">name</span><span class="p">::</span><span class="n">String</span>
    <span class="n">grad</span><span class="p">::</span><span class="kt">Float64</span>
    <span class="n">parents</span><span class="p">::</span><span class="n">Array</span><span class="p">{</span><span class="n">Any</span><span class="p">,</span><span class="mi">1</span><span class="p">}

    <span class="n">ADRev</span><span class="p">(</span><span class="n">val</span><span class="p">::</span><span class="kt">Float64</span><span class="p">,</span> <span class="n">name</span><span class="p">::</span><span class="n">String</span><span class="p">)</span> <span class="o">=</span> <span class="nb">new</span><span class="p">(</span><span class="n">val</span><span class="p">,</span> <span class="n">name</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">Any</span><span class="p">[])</span>
<span class="k">end</span>
</pre></div>

The replacement succeeded and the remainder appended. Now I'll convert the last post, `2016-11-21-AutoDiffNN.markdown`, in the same way and then run a repo-wide check for any remaining `{{ site.baseurl }}` usages to make them root-relative.
