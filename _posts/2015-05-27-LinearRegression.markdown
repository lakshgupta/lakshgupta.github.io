---
layout:     post
title:      "Linear Regression"
subtitle:   "Fundamentals of Neural Network 2"
date:       2015-05-27 12:00:00
author:     "Laksh Gupta"
header-img: "img/maggie-bg.jpg"
---

<p>
Alright, in the last <ahref src="http://lakshgupta.github.io/2015/05/21/ArtificialNeuron/">post</a> we looked at the very basic building block of a neural network: a neuron. But what could possibly a single neuron be 
good for? Well, as I mentioned in my last post it can be used to learn very simple models. Let us try to solve a linear regression problem using a neuron. 

<blockquote>
Linear regression is the simplest form of regression.  We model our system with a linear combination of features to produce one output.
<p align="right">- <a href="http://briandolhansky.com/blog/artificial-neural-networks-linear-regression-part-1">Brian Dolhansky</a></p>
</blockquote>
</p>

<h2 class="section-heading">The Problem</h2>
<p>I'll use the problem used in the Andrew Ng's machine learning course. We will try to predict the profit for the franchise based on the population of the city. We'll use the previous data to prepare a model. So let us first understand the data.</p>

</br><center><canvas id="inputData" width="600" height="400"></canvas></center></br>

<p>Looking at the data we can say that we don't need a complex model and linear regression is good enough for our purpose. </p>

<h2 class="section-heading">Training a model</h2>
</br></br><center><canvas id="artificialneuron" width="500" heigth="400"></canvas></center></br>
<p>
Our neuron will receive two values as an input. One of them is the actual value from the data and the other is a bias value. We usually include the bias 
value along with the input feature matrix x.
<blockquote>
b is the bias, a term that shifts the decision boundary away from the origin and does not depend on any input value.
<p align="right">- <a href="http://en.wikipedia.org/wiki/Perceptron">Wikipedia</a></p>
</blockquote>

Since we want to linearly fit the data, we'll use the linear activation function. When our neuron
will receive the inputs, we'll calculate the weighted sum and consider that as our output from the neuron.
$$f(x_i,w) = \phi(\sum\limits_{j=0}^n(w^j x_i^j)) = \sum\limits_{j=0}^n(w^j x_i^j) = w^Tx_i$$
where 
</p>

<p>
<ul>
<li> i represents a row of a matrix</li>
<li> j represetns an element of a matrix</li>
</ul>
</p>

<p>
The other way to look at our setup is that we are trying to fit a line to the data represented as
</br>$$y_i = w^0x_i^0 + w^1b$$</br>
</p>
<p>
We then try to figure out how close our neuron output or prediction is from the actual answer, i.e. we'll apply a <a href src="http://en.wikipedia.org/wiki/Loss_function">loss function</a> over our dataset. A commonly
used one is the least square error:
</br>$$L(w) = \sum\limits_{i=0}^n(f(x_i,w) - y_i)^2$$</br>
The idea is to use this value to modify our randomly initialized weight matrix till the time we stop observing the decrease in the loss function value.
The method we'll use to modify the weight matrix is know as <a href src="http://en.wikipedia.org/wiki/Gradient_descent">Gradient Descent</a>.
</p>



<!-- ############# JAVASCRIPT ############-->
<script language="javascript" type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/numeric/1.2.6/numeric.js" charset="utf-8"></script>
<script language="javascript" type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.4.4/p5.min.js" charset="utf-8"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/plot/scatter.js" charset="utf-8"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/utils/mathUtils.js" charset="utf-8"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/nn/canvas.js"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/nn/neuron.js"></script>
<script language="javascript" type="text/javascript" src="{{ site.baseurl }}/js/nn/neuralnet.js"></script>

<script language="javascript"> 
    
  //artificial neuron: linear model
  var _ancanvas = document.getElementById("artificialneuron");
  var _anctx = _ancanvas.getContext("2d");
  var neuronIn1 = new neuron(_anctx, 50, 40, neuronRadius,"b");
  var neuronIn2 = new neuron(_anctx, 50, 110, neuronRadius, "x_i^j");
  var	hiddenLayer= new neuron(_anctx, 200, 75, neuronRadius);
  _anctx.mathText("f(x_i, w)",200,120,{"text-align": "center"});
  var neuronOut = new neuron(_anctx, 350, 75, neuronRadius,"y");
  //input to hidden layer
  connectLayers([neuronIn1, neuronIn2], [hiddenLayer]);
  //hidden to output layer
  connectLayers([hiddenLayer], [neuronOut]);
  
  
  function setup(){
    loadTable("{{ site.baseurl }}/data/ex1data1.txt","CSV",linReg);
  }
  
  function linReg(table){
    var rowCount = table.rows.length - 1;
    var X = Array.matrix(rowCount, 2, 0);
    var Y = Array.matrix(rowCount, 1, 0);
    var iterations = 1500;
    var alpha = 0.01;
    //var theta = numeric.random([2,1]);
    var theta = Array.matrix(2,1,0);
    var xMax = table.getNum(0,0);
    var xMin = table.getNum(0,0);
    var yMax = table.getNum(0,1);
    var yMin = table.getNum(0,1);
    //load X and Y from table
    for(var i=0; i<rowCount; i++){
      X[i][0] = table.getNum(i,0);
      X[i][1] = 1;
      Y[i][0] = table.getNum(i,1);
      //find min and max
      if(xMax < X[i][0]){
        xMax = X[i][0];
      }
      if(xMin > X[i][0]){
        xMin = X[i][0];
      }
      if(yMax < Y[i][0]){
        yMax = Y[i][0];
      }
      if(yMin > Y[i][0]){
        yMin = Y[i][0];
      }
    }
    //plot input data
    var chartInfo= { y:{min:yMin, max:yMax, steps:5,label:"Profit in $10,000s"},
                      x:{min:xMin, max:xMax, steps:5,label:"Population of City in 10,000s"}
    };
    var inputPlot = new scatter("inputData",chartInfo, X, Y);
        
    //compute initial cost
    console.log(computeCost(X,Y, theta));
    
    //run gradient descent
    
    //plot the linear fit
    
    //predict the values
  }
  
  function computeCost(x,y, theta){
    var m = 1;
    if(Array.isArray(x)){
      m = x.length;
    } 
    return numeric.sum(numeric.pow(numeric.sub(numeric.dot(x, theta), y),2))/(2*m);
  };
</script>