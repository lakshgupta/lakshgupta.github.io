---
layout:     post
title:      "Automatic Differentiation using Operator Overloading"
subtitle:   "Automatic differentiation, derivatives"
date:       2016-11-06 12:00:00
author:     "Laksh Gupta"
header-img: "img/deathvalley1-bg.jpg"
---
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Almost all of the libraries for creating neural networks (Tensorflow, Theano, Torch, etc) are using automatic differentiation (AD) in one way or another. It has applications in the other parts of the mathematical world as well since it is a clever and effective way to calculate the gradients, effortlessly. It works by first creating a computational graph of the operations and then traversing it in either forward mode or reverse mode. Let us see how to implement them using operator overloading to calculate the first order partial derivative. I highly recommned reading Colah's blog <a href="http://colah.github.io/posts/2015-08-Backprop/">here</a> first. It has an excellent explanation about computational graphs and this post is related to the implementation side of it. It may not be the best performing piece of code for AD but I think it's the simplest one for getting your head around the concept. The example function we are considering here is:</p>
$$f(a, b) = (a + b) * (b + 1)$$<p>Using basic differential calculas rules, we can calculate the derivative of the above function with respect to a and b by applying the <a href="https://en.wikipedia.org/wiki/Sum_rule_in_differentiation">sum rule</a> and the <a href="https://en.wikipedia.org/wiki/Product_rule">product rule</a>:</p>
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
<p>The graph here can be thought to be constructed by way a programming langauge may perform the operations, using the <a href="https://en.wikipedia.org/wiki/Order_of_operations">BODMAS</a> rule. In terms of simple operations, the above function can be broken down to:</p>
$$c = a + b$$$$d = b + 1$$$$e = c * d$$<p>hence the operations to calculate the partial derivative of the above function with respect to a may look like:</p>
\begin{array}{cc|lcr|lcr}
\mathrm{value} && \mathrm{derivative} && node\\
\hline \\
a=a   && \frac{\partial a}{\partial a} = 1  && node 1\\
b=b   && \frac{\partial b}{\partial a} = 0  && node 2\\
c=a+b && \frac{\partial c}{\partial a} = \frac{\partial a}{\partial a} + \frac{\partial b}{\partial a}  && node3 \Leftarrow node1 + node2 \\
d=b+1 && \frac{\partial d}{\partial a} = \frac{\partial b}{\partial a} + \frac{\partial 1}{\partial a} && node5 \Leftarrow node2 + node4  \\
e=c*d && \frac{\partial e}{\partial a} = c*\frac{\partial d}{\partial a} + \frac{\partial c}{\partial a}*d && node6 \Leftarrow node3*node5 \\
\end{array}<p>To simulate the above steps, we have a type ADFwd which also represents a node in the calculation graph.</p>

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
<p>Now lets get the partial derivative of the above function with respect to 'a'.</p>

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


<div class="output_area"><div class="prompt output_prompt">Out[6]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>6.0</pre>
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
<div class=" highlight hl-julia"><pre><span></span><span class="c"># calculated derivative: de/da</span>
<span class="n">eaFwd</span><span class="o">.</span><span class="n">derivative</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[7]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>2.0</pre>
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
<p>Similarly, for 'b'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># define variables</span>
<span class="n">aFwd</span> <span class="o">=</span> <span class="n">ADFwd</span><span class="p">(</span><span class="mf">2.0</span><span class="p">)</span>
<span class="n">bFwd</span> <span class="o">=</span> <span class="n">ADFwd</span><span class="p">(</span><span class="mf">1.0</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[8]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>ADFwd(1.0,1.0)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># forward mode AD</span>
<span class="n">ebFwd</span> <span class="o">=</span> <span class="n">f</span><span class="p">(</span><span class="n">aFwd</span><span class="p">,</span> <span class="n">bFwd</span><span class="p">)</span>
<span class="n">ebFwd</span><span class="o">.</span><span class="n">value</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[9]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>6.0</pre>
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
<div class=" highlight hl-julia"><pre><span></span><span class="c"># calculated derivative: de/db</span>
<span class="n">ebFwd</span><span class="o">.</span><span class="n">derivative</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[10]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>5.0</pre>
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
<p>The partial derivative result will be present in the output ADFwd type variable. It represents the change in the output dependent variable with respect to the change in the input independent variable. The forward mode is simple to implement and does not take much memory. But if we have to calculate the derivative with resepect to multiple variables then we need to do the  forward pass for each variable . In such cases, reverse mode AD proves useful.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">Reverse Mode</h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Reverse mode helps in understanding the change in the inputs with respect to the change in the output. The first half of the reverse made is similar to the calculations as in the forward mode, we just don't calculate the derivatives. We move forward in the graph calculating the actual value of the sub-expression and then on reaching the output node, we set the output dependent variable's derivative component as 1. We use this derivative component along with the actual values calculated in the forward pass to apply the chain rule and calculate the derivative components for the parent dependent variable(s) and so on until the independent variables are reached.</p>
<blockquote><p>In a pen-and-paper calculation, one can perform the equivalent by repeatedly substituting the derivative of the outer functions in the chain rule:</p>
$${\displaystyle {\frac {\partial y}{\partial x}}={\frac {\partial y}{\partial w_{1}}}{\frac {\partial w_{1}}{\partial x}}=\left({\frac {\partial y}{\partial w_{2}}}{\frac {\partial w_{2}}{\partial w_{1}}}\right){\frac {\partial w_{1}}{\partial x}}=\left(\left({\frac {\partial y}{\partial w_{3}}}{\frac {\partial w_{3}}{\partial w_{2}}}\right){\frac {\partial w_{2}}{\partial w_{1}}}\right){\frac {\partial w_{1}}{\partial x}}=\cdots }$$<p>- <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">wikipedia</a></p>
</blockquote>
<p>We can see the equations during the reverse pass as:</p>
\begin{array}{cc}
\mathrm{derivative} && child node \Leftarrow parent node\\
\hline \\
\frac{\partial e}{\partial e} = 1  && node6\\
\frac{\partial e}{\partial c} = \frac{\partial e}{\partial e}*\frac{\partial e}{\partial c} = 1*d && node 3 \Leftarrow node 6\\
\frac{\partial e}{\partial d} = \frac{\partial e}{\partial e}*\frac{\partial e}{\partial d} = 1*c && node5 \Leftarrow node6 \\
\frac{\partial e}{\partial a} = \frac{\partial e}{\partial c}*\frac{\partial c}{\partial a} = d*1 && node1 \Leftarrow node3 \\
\frac{\partial e}{\partial b} = \frac{\partial e}{\partial c}*\frac{\partial c}{\partial b} + \frac{\partial e}{\partial d}*\frac{\partial d}{\partial b} =  d*1 + c*1 && node2 \Leftarrow node3,node5 \\
\end{array}<p>In the implementation, we have a type ADRev which stores the value and the derivative for a particular node. We also store the parents during the forward pass to propagate the derivative backwards during the reverse pass.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># type to store the float value for a variable during the forward pass</span>
<span class="c"># and the derivative during the reverse pass. </span>
<span class="k">type</span><span class="nc"> ADRev</span>
    <span class="n">value</span><span class="p">::</span><span class="kt">Float64</span>
    <span class="n">derivative</span><span class="p">::</span><span class="kt">Float64</span>
    <span class="n">derivativeOp</span><span class="p">::</span><span class="n">Function</span>
    <span class="n">parents</span><span class="p">::</span><span class="n">Array</span><span class="p">{</span><span class="n">ADRev</span><span class="p">}</span>
    
    <span class="n">ADRev</span><span class="p">(</span><span class="n">val</span><span class="p">::</span><span class="kt">Float64</span><span class="p">)</span> <span class="o">=</span> <span class="nb">new</span><span class="p">(</span><span class="n">val</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">ad_constD</span><span class="p">,</span> <span class="n">Array</span><span class="p">(</span><span class="n">ADRev</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
    <span class="n">ADRev</span><span class="p">(</span><span class="n">val</span><span class="p">::</span><span class="kt">Float64</span><span class="p">,</span> <span class="n">der</span><span class="p">::</span><span class="kt">Float64</span><span class="p">)</span> <span class="o">=</span> <span class="nb">new</span><span class="p">(</span><span class="n">val</span><span class="p">,</span> <span class="n">der</span><span class="p">,</span> <span class="n">ad_constD</span><span class="p">,</span> <span class="n">Array</span><span class="p">(</span><span class="n">ADRev</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="k">end</span>

<span class="k">function</span><span class="nf"> ad_constD</span><span class="p">(</span><span class="n">prevDerivative</span><span class="p">::</span><span class="kt">Float64</span><span class="p">,</span> <span class="n">adNodes</span><span class="p">::</span><span class="n">Array</span><span class="p">{</span><span class="n">ADRev</span><span class="p">})</span>
    <span class="k">return</span> <span class="mi">0</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[11]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>ad_constD (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># define the actual addition operation and the derivative rule to use</span>
<span class="c"># during the reverse pass.</span>
<span class="k">function</span><span class="nf"> adr_add</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADRev</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADRev</span><span class="p">)</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">ADRev</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">value</span> <span class="o">+</span> <span class="n">y</span><span class="o">.</span><span class="n">value</span><span class="p">)</span>
    <span class="n">result</span><span class="o">.</span><span class="n">derivativeOp</span> <span class="o">=</span> <span class="n">adr_addD</span>
    <span class="n">push!</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">parents</span><span class="p">,</span> <span class="n">x</span><span class="p">)</span>
    <span class="n">push!</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">parents</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">result</span>
<span class="k">end</span>
<span class="k">function</span><span class="nf"> adr_addD</span><span class="p">(</span><span class="n">prevDerivative</span><span class="p">::</span><span class="kt">Float64</span><span class="p">,</span> <span class="n">adNodes</span><span class="p">::</span><span class="n">Array</span><span class="p">{</span><span class="n">ADRev</span><span class="p">})</span>
    <span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">=</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="n">prevDerivative</span> <span class="o">*</span> <span class="mi">1</span>
    <span class="n">adNodes</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">=</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="n">prevDerivative</span> <span class="o">*</span> <span class="mi">1</span>
    <span class="k">return</span>
<span class="k">end</span>
<span class="o">+</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADRev</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADRev</span><span class="p">)</span> <span class="o">=</span> <span class="n">adr_add</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[12]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>+ (generic function with 165 methods)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># define the actual multiplication operation and the derivative rule to use</span>
<span class="c"># during the reverse pass.</span>
<span class="k">function</span><span class="nf"> adr_mul</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADRev</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADRev</span><span class="p">)</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">ADRev</span><span class="p">(</span><span class="n">x</span><span class="o">.</span><span class="n">value</span> <span class="o">*</span> <span class="n">y</span><span class="o">.</span><span class="n">value</span><span class="p">)</span>
    <span class="n">result</span><span class="o">.</span><span class="n">derivativeOp</span> <span class="o">=</span> <span class="n">adr_mulD</span>
    <span class="n">push!</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">parents</span><span class="p">,</span> <span class="n">x</span><span class="p">)</span>
    <span class="n">push!</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">parents</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">result</span>
<span class="k">end</span>
<span class="k">function</span><span class="nf"> adr_mulD</span><span class="p">(</span><span class="n">prevDerivative</span><span class="p">::</span><span class="kt">Float64</span><span class="p">,</span> <span class="n">adNodes</span><span class="p">::</span><span class="n">Array</span><span class="p">{</span><span class="n">ADRev</span><span class="p">})</span>
    <span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">=</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="n">prevDerivative</span> <span class="o">*</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">value</span>
    <span class="n">adNodes</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">=</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="n">prevDerivative</span> <span class="o">*</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">value</span>
    <span class="k">return</span>
<span class="k">end</span>
<span class="o">*</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADRev</span><span class="p">,</span> <span class="n">y</span><span class="p">::</span><span class="n">ADRev</span><span class="p">)</span> <span class="o">=</span> <span class="n">adr_mul</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[13]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>* (generic function with 151 methods)</pre>
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
<p>We are doing a breadth-first graph traversal to propagate the derivates backward during the reverse pass. Since the objects are passed using reference, updating the parent having multiple children becomes trivial. For example, node 2 needs to accumulate the derivate from node 3 and node 5 in our case, both of which may get evaluated separately during the traversal. And this is why we are adding the calculated derivative instead of directly assigning it to the node's derivative.</p>
<div class="highlight"><pre><span></span><span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">=</span> <span class="n">adNodes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">derivative</span> <span class="o">+</span> <span class="o">...</span>
</pre></div>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># this is the reverse pass where we apply the chain rule</span>
<span class="k">function</span><span class="nf"> chainRule</span><span class="p">(</span><span class="n">graph</span><span class="p">::</span><span class="n">ADRev</span><span class="p">)</span>
    <span class="n">current</span> <span class="o">=</span> <span class="n">graph</span>
    <span class="c"># set the derivative to 1</span>
    <span class="n">current</span><span class="o">.</span><span class="n">derivative</span> <span class="o">=</span> <span class="mi">1</span>
    <span class="n">bfs</span> <span class="o">=</span> <span class="p">[</span><span class="n">current</span><span class="p">]</span>
    <span class="k">while</span> <span class="n">length</span><span class="p">(</span><span class="n">bfs</span><span class="p">)</span> <span class="o">!=</span> <span class="mi">0</span>
        <span class="n">current</span> <span class="o">=</span> <span class="n">pop!</span><span class="p">(</span><span class="n">bfs</span><span class="p">)</span>
        <span class="n">currDerivative</span> <span class="o">=</span> <span class="n">current</span><span class="o">.</span><span class="n">derivative</span>
        <span class="n">current</span><span class="o">.</span><span class="n">derivativeOp</span><span class="p">(</span><span class="n">currDerivative</span><span class="p">,</span> <span class="n">current</span><span class="o">.</span><span class="n">parents</span><span class="p">)</span>
        <span class="n">numParents</span> <span class="o">=</span> <span class="n">length</span><span class="p">(</span><span class="n">current</span><span class="o">.</span><span class="n">parents</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">i</span><span class="o">=</span><span class="mi">1</span><span class="p">:</span><span class="n">numParents</span> 
            <span class="n">push!</span><span class="p">(</span><span class="n">bfs</span><span class="p">,</span> <span class="n">current</span><span class="o">.</span><span class="n">parents</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
        <span class="k">end</span>
    <span class="k">end</span>
    <span class="k">return</span> <span class="n">graph</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[14]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>chainRule (generic function with 1 method)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># define the function</span>
<span class="k">function</span><span class="nf"> f</span><span class="p">(</span><span class="n">x</span><span class="p">::</span><span class="n">ADRev</span><span class="p">,</span><span class="n">y</span><span class="p">::</span><span class="n">ADRev</span><span class="p">)</span>
    <span class="p">(</span><span class="n">x</span><span class="o">+</span><span class="n">y</span><span class="p">)</span><span class="o">*</span><span class="p">(</span><span class="n">y</span> <span class="o">+</span> <span class="n">ADRev</span><span class="p">(</span><span class="mf">1.0</span><span class="p">))</span>
<span class="k">end</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[15]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>f (generic function with 2 methods)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># create the variables</span>
<span class="n">aRev</span> <span class="o">=</span> <span class="n">ADRev</span><span class="p">(</span><span class="mf">2.0</span><span class="p">)</span>
<span class="n">bRev</span> <span class="o">=</span> <span class="n">ADRev</span><span class="p">(</span><span class="mf">1.0</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[16]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>ADRev(1.0,0.0,ad_constD,ADRev[])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># forward pass</span>
<span class="n">eRev_forward</span> <span class="o">=</span> <span class="n">f</span><span class="p">(</span><span class="n">aRev</span><span class="p">,</span> <span class="n">bRev</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[17]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>ADRev(6.0,0.0,adr_mulD,ADRev[ADRev(3.0,0.0,adr_addD,ADRev[ADRev(2.0,0.0,ad_constD,ADRev[]),ADRev(1.0,0.0,ad_constD,ADRev[])]),ADRev(2.0,0.0,adr_addD,ADRev[ADRev(1.0,0.0,ad_constD,ADRev[]),ADRev(1.0,0.0,ad_constD,ADRev[])])])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># reverse pass</span>
<span class="n">eRev_reverse</span> <span class="o">=</span> <span class="n">chainRule</span><span class="p">(</span><span class="n">eRev_forward</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[18]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>ADRev(6.0,1.0,adr_mulD,ADRev[ADRev(3.0,2.0,adr_addD,ADRev[ADRev(2.0,2.0,ad_constD,ADRev[]),ADRev(1.0,5.0,ad_constD,ADRev[])]),ADRev(2.0,3.0,adr_addD,ADRev[ADRev(1.0,5.0,ad_constD,ADRev[]),ADRev(1.0,3.0,ad_constD,ADRev[])])])</pre>
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
<p>Since we are storing the graph during the forward pass to help us in the reverse pass, the output variable can explain the parent-child relationship as well as the operations performed on each of the nodes.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span></span><span class="c"># derivative with respect to all the independent variables</span>
<span class="n">aRev</span><span class="o">.</span><span class="n">derivative</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[19]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>2.0</pre>
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
<div class=" highlight hl-julia"><pre><span></span><span class="n">bRev</span><span class="o">.</span><span class="n">derivative</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[20]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>5.0</pre>
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
<p>As mentioned before, the benefit of using reverse mode AD is that we can calculate the derivative of the output with respect to each of the input variables in a single iteration only. We'll use this property to implement a neural network in the coming post.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 class="section-heading">References:</h2><ul>
<li><a href="https://en.wikipedia.org/wiki/Automatic_differentiation">Automatic differentiation</a></li>
<li><a href="http://colah.github.io/posts/2015-08-Backprop/">Calculus on Computational Graphs: Backpropagation</a></li>
<li><a href="http://blog.tombowles.me.uk/2014/09/10/ad-algorithmicautomatic-differentiation/">ALGORITHMIC/AUTOMATIC DIFFERENTIATION</a></li>
<li><a href="http://www.win-vector.com/dfiles/ReverseAccumulation.pdf">Gradients via Reverse Accumulation</a></li>
<li><a href="http://stats.stackexchange.com/questions/224140/step-by-step-example-of-reverse-mode-automatic-differentiation">Step-by-step example of reverse-mode automatic differentiation</a></li>
<li><a href="https://www.duo.uio.no/bitstream/handle/10852/41535/Kjelseth-Master.pdf?sequence=9">Efficient Calculation of Derivatives using Automatic Differentiation</a></li>
</ul>

</div>
</div>
</div>
    </div>
  </div>