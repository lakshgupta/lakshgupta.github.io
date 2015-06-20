---
layout:     post
title:      "Vector Representation of Words"
subtitle:   "text data as an input"
date:       2015-06-12 12:00:00
author:     "Laksh Gupta"
header-img: "img/sd3-bg.jpg"
---

Before moving on to see how we can connect several neurons together to make a more powerful model of a brain, let us see how we can process the textual information to create a vector representation, also known as word embeddings or word vectors, which can be used as an input to a neural network. 

<h2 class="section-heading">One-Hot Vector</h2>
This is the most simplest one where for each word we create a vector of length equal to the size of the vocabulary, $$R^{\left\|V\right\|}$$. We fill the vector with $$0's$$ and $$1's$$ at the index of the word. 

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
\end{bmatrix}
$$

All these vectors are independent to each other. Hence this representation doesn't encodes any relationship between words:

$$(W^{apple})^TW^{banana}=(W^{king})^TW^{queen}=0$$


<blockquote>
You shall know a word by the company it keeps (Firth, J. R. 1957:11)
<p align="right">- <a href="https://en.wikipedia.org/wiki/John_Rupert_Firth">Wikipedia</a></p>
</blockquote>

<h2 class="section-heading">Word-Document Matrix</h2>

<h2 class="section-heading">Word-Word Matrix</h2>

---

References:

- [From Frequency to Meaning: Vector Space Models of Semantics](http://arxiv.org/abs/1003.1141)

