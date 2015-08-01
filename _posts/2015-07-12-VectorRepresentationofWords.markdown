---
layout:     post
title:      "Vector Representation of Words"
subtitle:   "text data as an input"
date:       2015-07-12 12:00:00
author:     "Laksh Gupta"
header-img: "img/sd4-bg.jpg"
---

Before moving on to see how we can connect several neurons together to make a more powerful model of a brain, let us see how we can process the textual information to create a vector representation, also known as word embeddings or word vectors, which can be used as an input to a neural network. 

<h2 class="section-heading">One-Hot Vector</h2>
This is the most simplest one where for each word we create a vector of length equal to the size of the vocabulary, $$R^{\left\|V\right\|}$$. We fill the vector with $$1$$ at the index of the word, rest all $$0s$$. 

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

Also, each vector would be very sparse. Hence this approach requires large space to encode all our words in the vector form.

<blockquote>
You shall know a word by the company it keeps (Firth, J. R. 1957:11)
<p align="right">- <a href="https://en.wikipedia.org/wiki/John_Rupert_Firth">Wikipedia</a></p>
</blockquote>

<h2 class="section-heading">Word-Document Matrix</h2>

In this approach, we create a matrix where a column represents a document and a row represents the frequency of a word in the document. This matrix scales with the number of documents ($$D$$). The matrix size would be $$R^{\left\|D*V\right\|}$$ where $$V$$ is the size of the vocabulary.

<h2 class="section-heading">Word-Word Matrix</h2>

In this case, we build a co-occurence matrix where both columns and rows represent words from the vocabulary. The benefit of building this matrix is that the co-occurence value of the words which are highly likely to come together in a sentence will always be high as compared to the words which rarely come together. Hence we should be fine once we have a descent sized dataset or say documents. Also, the size of the matrix dependent now on the size of the vocabulary, $$R^{\left\|V*V\right\|}$$.

The beauty of the last two approaches is that we can apply [Singular-Value-Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) on the matrix and further reduce the dimentionality. Let us see an example on the Word-Word matrix.

<div>
<iframe height="500" src="{{ site.baseurl }}/notebooks/wordVec_SVD.html"></iframe></br>
</div>

<h2 class="section-heading">Continuous Bag of Words Model (CBOW)</h2>

<h2 class="section-heading">Skip-Gram Model</h2>


---

References:

- [From Frequency to Meaning: Vector Space Models of Semantics](http://arxiv.org/abs/1003.1141)
- [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781)
- [Singular Value Decomposition Tutorial PDF](https://www.ling.ohio-state.edu/~kbaker/pubs/Singular_Value_Decomposition_Tutorial.pdf)
- [Dimensionality Reduction](http://infolab.stanford.edu/~ullman/mmds/ch11.pdf)

<script language="javascript"> 

</script>