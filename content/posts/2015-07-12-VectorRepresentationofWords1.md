---
title: "Vector Representation of Words - 1"
subtitle: "text data as an input"
description: "Word embeddings and vector representations: one-hot vectors, co-occurrence matrices, and SVD."
date: 2015-07-12T12:00:00
author: "Laksh Gupta"
tags: ["nlp", "word-embeddings", "machine-learning"]
---
 <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let us see how we can process the textual information to create a vector representation, also known as word embeddings or word vectors, which can be used as an input to a neural network.</p>
<h2 class="section-heading">One-Hot Vector</h2><p>This is the most simplest one where for each word we create a vector of length equal to the size of the vocabulary, $R^{\left\|V\right\|}$. We fill the vector with $1$ at the index of the word, rest all $0s$.</p>
$$W^{apple} = 
\begin{bmatrix}
  1 \\ 
  \vdots \\
  \vdots \\
  \vdots \\
  0 \\
\end{bmatrix}
W^{banana} = 
\begin{bmatrix}
  0 \\ 
  1 \\
  \vdots \\
  \vdots \\
  0 \\
\end{bmatrix}
W^{king} = 
\begin{bmatrix}
  0 \\ 
  \vdots \\
  1 \\
  \vdots \\
  0 \\
\end{bmatrix}
W^{queen} = 
\begin{bmatrix}
  0 \\ 
  \vdots \\
  \vdots \\
  1 \\
  0 \\
\end{bmatrix}$$<p>All these vectors are independent to each other. Hence this representation doesn't encodes any relationship between words:</p>
$$(W^{apple})^TW^{banana}=(W^{king})^TW^{queen}=0$$<p>Also, each vector would be very sparse. Hence this approach requires large space to encode all our words in the vector form.</p>
<blockquote>
You shall know a word by the company it keeps (Firth, J. R. 1957:11)
<p align="right">- <a href="https://en.wikipedia.org/wiki/John_Rupert_Firth">Wikipedia</a></p>
</blockquote><h2 class="section-heading">Word-Document Matrix</h2><p>In this approach, we create a matrix where a column represents a document and a row represents the frequency of a word in the document. This matrix scales with the number of documents ($D$). The matrix size would be $R^{\left\|D*V\right\|}$ where $V$ is the size of the vocabulary.</p>
<h2 class="section-heading">Word-Word Matrix</h2><p>In this case, we build a co-occurence matrix where both columns and rows represent words from the vocabulary. The benefit of building this matrix is that the co-occurence value of the words which are highly likely to come together in a sentence will always be high as compared to the words which rarely come together. Hence we should be fine once we have a descent sized dataset or say documents. Also, the size of the matrix dependent now on the size of the vocabulary, $R^{\left\|V*V\right\|}$.</p>
<p>The beauty of the last two approaches is that we can apply <a href="https://en.wikipedia.org/wiki/Singular_value_decomposition">Singular-Value-Decomposition</a> (SVD) on the matrix and further reduce the dimentionality. Let us see an example on the Word-Word matrix.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Consider our data to have the following 3 sentence:</p>
<ul>
<li>I enjoy driving.</li>
<li>I like banana.</li>
<li>I like reading.</li>
</ul>
<p>The co-occurence matrix will look like:</p>
$$X = 
\begin{array}{c|lcr}
words & \text{I} & \text{enjoy} & \text{driving} & \text{like} & \text{banana} & \text{reading} &\text{.}\\
\hline
\text{I} & 0 & 1 & 0 & 2 & 0 & 0 & 0 \\
\text{enjoy} & 1 & 0 & 1 & 0 & 0 & 0 & 0 \\
\text{driving} & 0 & 1 & 0 & 0 & 0 & 0 & 1 \\
\text{like} & 2 & 0 & 0 & 0 & 1 & 1 & 0 \\
\text{banana} & 0 & 0 & 0 & 1 & 0 & 0 & 1 \\
\text{reading} & 0 & 0 & 0 & 1 & 0 & 0 & 1 \\
\text{.} & 0 & 0 & 1 & 0 & 1 & 1 & 0 \\
\end{array}
$$
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="n">words</span> <span class="o">=</span> <span class="p">[</span><span class="s">"I"</span> <span class="s">"enjoy"</span> <span class="s">"driving"</span> <span class="s">"like"</span> <span class="s">"banana"</span> <span class="s">"reading"</span> <span class="s">"."</span><span class="p">];</span>
<span class="n">X</span> <span class="o">=</span>   <span class="p">[</span><span class="mi">0</span> <span class="mi">1</span> <span class="mi">0</span> <span class="mi">2</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span><span class="p">;</span>
       <span class="mi">1</span> <span class="mi">0</span> <span class="mi">1</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span><span class="p">;</span>
       <span class="mi">0</span> <span class="mi">1</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span><span class="p">;</span>
       <span class="mi">2</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span> <span class="mi">1</span> <span class="mi">0</span><span class="p">;</span>
       <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span><span class="p">;</span>
       <span class="mi">0</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span> <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span>
       <span class="mi">0</span> <span class="mi">0</span> <span class="mi">1</span> <span class="mi">0</span> <span class="mi">1</span> <span class="mi">1</span> <span class="mi">0</span><span class="p">];</span>
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
<p>In <a href="http://julia.readthedocs.org/en/latest/stdlib/linalg/">Julia</a>, applying SVD on our matrix $X$ will give us $U$, $S$ and $V$ where:</p>
<center>$$A == U*diagm(S)*V^T$$</center>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="n">U</span><span class="p">,</span><span class="n">S</span><span class="p">,</span><span class="n">V</span> <span class="o">=</span> <span class="n">svd</span><span class="p">(</span><span class="n">X</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-julia"><pre><span class="n">U</span>
</pre></div>

</div>
</div>
</div>

**/