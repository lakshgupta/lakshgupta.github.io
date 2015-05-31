---
layout:     post
title:      "Artificial Neuron"
subtitle:   "Fundamentals of Neural Network 1"
date:       2015-05-21 12:00:00
author:     "Laksh Gupta"
header-img: "img/sd1-bg.jpg"
---

Billions of neuron work together in a highly parallel manner to form the most sophisticated computing device known as the human brain. A single neuron:

- receives the electrical signals from the axons of other neurons through dendrites. Signals can come from different organs such as eyes and ears.
- modulates the signals in various amounts at the synapses between the dendrite and axons.
- fires an output signal only when the total strength of the input signals exceed a certain threshold. This signal is sent further to other neurons.

<p></br></p>
<center>
<img src="{{ site.baseurl }}/img/nn/bioneuron.jpg" alt="Biological Neuron">
<span class="caption text-muted">A Biological Neuron</span>
</center>

More information about a biological neuron can be found on <a href="http://en.wikipedia.org/wiki/Neuron">Wikipedia</a>.

<h2 class="section-heading">Artificial Neuron</h2>
An artificial neuron is a mathematical model of a biological neuron. The steps mentined for a biological neuron can be mapped to an artificial neuron as:

- an artificial neuron receives the input as numerical values rather than the electrical signals. Input can come from different sources such as an image or a text.
- it then multiplies each of the input value by a value called the weight.
- weighted sum is calculated then to represent the total strength of the input signal, and an activation function is applied on the sum to get the output. This output can be sent further to other artificial neurons.



<blockquote>
The artificial neuron receives one or more inputs (representing dendrites) and sums them to produce an output (representing a neuron's axon). Usually the sums of each node are weighted, and the sum is passed through a non-linear function known as an activation function or transfer function. The transfer functions usually have a sigmoid shape, but they may also take the form of other non-linear functions, piecewise linear functions, or step functions. They are also often monotonically increasing, continuous, differentiable and bounded.
<p align="right">- <a href="http://en.wikipedia.org/wiki/Artificial_neuron">Wikipedia</a></p>
</blockquote>


<center><canvas id="artificialneuron" width="500" heigth="400"></canvas></center>


- $$f(w^Tx) = \phi(\sum\limits_{i=0}^n(w_i x_i))$$ &nbsp;
- $$\phi$$ is our activation function.
- $$x_i$$ are the elements of the input matrix x.
- $$w_i$$ are the elements of the weight matrix y. 
- $$y$$ is the output.


Notice that in terms of "learning" in almost all of the machine learning algorithms, we learn the weight parameters $$w_i$$. 

<h2 class="section-heading">Activation Function</h2>
An artificial neuron using a step activation function is known as a Perceptron. Perceptron can act as a binary classifier based on if the value of the activation function is above or below a threashold. But step activation function may not be a good choice every time.


<blockquote>
  In fact, a small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0 to 1. That flip may then cause the behaviour of the rest of the network to completely change in some very complicated way. So while your "9" might now be classified correctly, the behaviour of the network on all the other images is likely to have completely changed in some hard-to-control way. That makes it difficult to see how to gradually modify the weights and biases so that the network gets closer to the desired behaviour.
  <p align="right">- <a href="http://neuralnetworksanddeeplearning.com/chap1.html">Michael Nielsen</a></p>
</blockquote>


There are other activation functions which seem to work generally better in most of the cases, such as tanh or maxout functions.


<blockquote>
  "What neuron type should I use?" Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of "dead" units in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work worse than ReLU/Maxout.
  <p align="right">- <a href="http://cs231n.github.io/neural-networks-1/">Andrej Karpathy</a></p>
</blockquote>


<center>
 <canvas id="step" width="200" height="200"></canvas>
 <canvas id="sigmoid" width="200" height="200"></canvas>
 <canvas id="tanh" width="200" height="200"></canvas></br>
</center>

Other than linear classification, we can also perform linear regression using a single neuron. I'll try to cover the implementation of both of them in the coming posts. Till then, enjoy!

<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/nn/canvas.js"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/nn/neuron.js"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/nn/neuralnet.js"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/plot/eqgraph.js" charset="utf-8"></script>
<script>
//artificial neuron
var _ancanvas = document.getElementById("artificialneuron");
var _anctx = _ancanvas.getContext("2d");
var neuronIn1 = new neuron(_anctx, 50, 40, neuronRadius,"x_0");
var neuronIn2 = new neuron(_anctx, 50, 110, neuronRadius, "x_n");
var	hiddenLayer= new neuron(_anctx, 200, 75, neuronRadius);
_anctx.mathText("f(w^Tx)",200,120,{"text-align": "center"});
var neuronOut = new neuron(_anctx, 350, 75, neuronRadius,"y");
//input to hidden layer
connectLayers([neuronIn1, neuronIn2], [hiddenLayer]);
//hidden to output layer
connectLayers([hiddenLayer], [neuronOut]);

//plot step
function step(z){ 
        if(z < 2){
          return 0;
        }else{
          return 1;
        }
      }
var stepGraph = new EqGraph({canvasId: 'step', minX: -4, minY: -2, maxX: 4, maxY: 2, unitsPerTick: 1 });
stepGraph.drawEquation(step , 'blue', 2);
var stepCanv = document.getElementById('step');
var stepcontext = stepCanv.getContext('2d');
stepcontext.font = 'italic 14pt Calibri';
stepcontext.fillStyle = '#777';
stepcontext.fillText('step', 10, stepCanv.height-5);

//plot sigmoid
function sigmoid(z){ return  1.0/(1.0+Math.exp(-z));}
var sigmoidGraph = new EqGraph({canvasId: 'sigmoid', minX: -6, minY: -2, maxX: 6, maxY: 2, unitsPerTick: 1 });
sigmoidGraph.drawEquation(sigmoid , 'blue', 2);
var sigmoidCanv = document.getElementById('sigmoid');
var sigmoidcontext = sigmoidCanv.getContext('2d');
sigmoidcontext.font = 'italic 14pt Calibri';
sigmoidcontext.fillStyle = '#777';
sigmoidcontext.fillText('sigmoid', 10, sigmoidCanv.height-5);

//plot tanh
function tanh(z){ return (Math.exp(z)-Math.exp(-z))/(Math.exp(z)+Math.exp(-z));}
var tanhGraph = new EqGraph({canvasId: 'tanh', minX: -6, minY: -2, maxX: 6, maxY: 2, unitsPerTick: 1 });
tanhGraph.drawEquation(tanh , 'blue', 2);
var tanhCanv = document.getElementById('tanh');
var tanhcontext = tanhCanv.getContext('2d');
tanhcontext.font = 'italic 14pt Calibri';
tanhcontext.fillStyle = '#777';
tanhcontext.fillText('tanh', 10, tanhCanv.height-5);

</script>
