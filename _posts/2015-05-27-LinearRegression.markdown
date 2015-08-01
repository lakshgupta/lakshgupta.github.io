---
layout:     post
title:      "Linear Regression"
subtitle:   "what can a neuron do"
date:       2015-05-27 12:00:00
author:     "Laksh Gupta"
header-img: "img/sd2-bg.jpg"
---


Alright, in the last [post](http://lakshgupta.github.io/2015/05/21/ArtificialNeuron/) we looked at the very basic building block of a neural network: a neuron. But what could possibly a single neuron be good for? Well, as I mentioned in my last post it can be used to learn very simple models. Let us try to solve a linear regression problem using a neuron.


<blockquote>
Linear regression is the simplest form of regression.  We model our system with a linear combination of features to produce one output.
<p align="right">- <a href="http://briandolhansky.com/blog/artificial-neural-networks-linear-regression-part-1">Brian Dolhansky</a></p>
</blockquote>


<h2 class="section-heading">The Problem</h2>


I'll use the problem used in the Andrew Ng's machine learning course. The dataset is located [here](https://github.com/lakshgupta/lakshgupta.github.io/blob/master/data/ex1data1.txt). We will try to predict the profit for the franchise based on the population of the city. We'll use the previous data to prepare a model. So let us first understand the data.


<center><canvas id="inputData" width="600" height="400"></canvas></center>


Looking at the data we can say that we don't need a complex model and linear regression is good enough for our purpose.


<h2 class="section-heading">Training a model</h2>


<center><canvas id="artificialneuron" width="500" height="150"></canvas></center>

Our neuron will receive two values as an input. One of them is the actual value from the data and the other is a bias value. We usually include the bias value along with the input feature matrix x.


<blockquote>
b is the bias, a term that shifts the decision boundary away from the origin and does not depend on any input value.
<p align="right">- <a href="http://en.wikipedia.org/wiki/Perceptron">Wikipedia</a></p>
</blockquote>

Since we want to linearly fit the data, we'll use the linear activation function. When our neuron will receive the inputs, we'll calculate the weighted sum and consider that as our output from the neuron.
<center>$$f(x_i,w) = \phi(\sum\limits_{j=0}^n(w^j x_i^j)) = \sum\limits_{j=0}^n(w^j x_i^j) = w^Tx_i$$</center>
where 

- $$i$$ represents a row of a matrix
- $$j$$ represetns an element of a matrix


The other way to look at our setup is that we are trying to fit a line to the data represented as

$$y_i = w^0x_i^0 + w^1b$$


We then try to figure out how close our neuron output or prediction is from the actual answer, i.e. we'll apply a <a href="http://en.wikipedia.org/wiki/Loss_function">loss function</a>, also known as a cost function over our dataset. A commonly used one is the least square error:
<center>$$J(w) = \sum\limits_{i=0}^n(f(x_i,w) - y_i)^2$$</center>
The idea is to use this value to modify our randomly initialized weight matrix till the time we stop observing the decrease in the cost function value. The method we'll use to modify the weight matrix is known as [Gradient Descent](http://en.wikipedia.org/wiki/Gradient_descent).
<center>$$w = w - \frac{\alpha}{m}\Delta J(w)$$</center>
here 

- $$w$$ is the weight matrix
- $$\alpha$$ is the learning rate
- $$m$$ is the size of our data acting as a normalizing factor
- $$\Delta J(w)$$ is the gradient of the cost function with respect to each of the weight under consideration say weight for the connection between a neuron $$j$$ and a neuron $$k$$


$$\frac{\partial}{\partial w_{jk}} J(w) = \sum\limits_{i=0}^n 2\left(f(x_i, w)-y_i\right) \frac{\partial}{\partial w_{jk}} f(x_i, w) $$


So let us train the model and see how it is behaving by plotting the results of the above equation in red using the weight matrix and the x-axis.
<center>
<canvas id="fitData" height="400px" width="600">
This text is displayed if your browser does not support HTML5 Canvas.
</canvas>
</center>


<h2 class="section-heading">Making a Prediction</h2>
To make a prediction we just need to use the modified weight matrix, obtained after the gradient descent step, along with the new input values and apply the same function we used above:
<center>$$f(x_i,w) = w^Tx_i$$</center>



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
  
  var iterations = 1000;//1500;
  var learningRate = 0.01;
  
  function setup(){
    loadTable("{{ site.baseurl }}/data/ex1data1.txt","CSV",linReg);
  }
  
  function linReg(table){
    var rowCount = table.rows.length - 1;
    var X = Array.matrix(rowCount, 2, 0);
    var Y = Array.matrix(rowCount, 1, 0);
    J_history = Array.matrix(iterations,1, 0);
    var m = X.length;
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
    console.log("Initial cost: "+ computeCost(X,Y, theta));
    console.log("Initial theta: "+theta);
    //run gradient descent
    for(var i=0;i<iterations;i++){
	    var tempTheta = theta;
	    //console.log("gradient descent theta: " +theta);
	    //for each weight
	    var subCorrection1 = numeric.sub(numeric.dot(X, tempTheta), Y);
	    //console.log("subCorrection1: "+subCorrection1);
	    
	    for (var j=0; j < theta.length; j++)
	    {
	        //console.log("weight: "+j);
	        //for each input row
	        var subCorrection2 = subCorrection1.slice(0);
	        for(var k=0;k<m;k++)
	        {
	          //console.log("before subCorrection2[k]: "+ subCorrection2[k]);
		        subCorrection2[k] = subCorrection2[k]*X[k][j];
		        //console.log("x val: "+ X[k][j]);
		        //console.log("after subCorrection2[k]: "+ subCorrection2[k]);
	        }
	        //console.log(subCorrection2);
	        correction = (learningRate/m) * numeric.sum(subCorrection2);
	        subCorrection2 = new Array();// = [];
	        //console.log("after haha subcorrection2: "+subCorrection2);
	        //console.log("after haha subcorrection1: "+subCorrection1);
		      //console.log("correction: "+correction);
		      theta[j] = theta[j] - correction;
      }
      //Save the cost J in every iteration    
      J_history[i] = computeCost(X, y, theta);
    }
    console.log("Cost history: "+J_history);
    console.log("Final theta: "+ theta);
    //plot the linear fit
    var fitChartInfo= { y:{min:yMin, max:yMax, steps:5,label:"Profit in $10,000s"},
                      x:{min:xMin, max:xMax, steps:5,label:"Population of City in 10,000s"}
    };
    var fitPlot = new scatter("fitData",fitChartInfo, X, Y);
    //console.log(theta);
    fitPlot.plotLine(theta);
    
  }
  
  
  //loss function
  function computeCost(x,y, theta){
    var m = 1;
    if(Array.isArray(x)){
      m = x.length;
    } 
    return numeric.sum(numeric.pow(numeric.sub(numeric.dot(x, theta), y),2));///(2*m);
  };
</script>