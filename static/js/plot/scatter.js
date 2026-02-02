/*
//defining the data
var data = [{label:"David",
math:50,
english:80,
art:92,
style:"rgba(241, 178, 225, 0.5)"},
{label:"Ben",
math:80,
english:60,
art:43,style:"#B1DDF3"},
{label:"Oren",
math:70,
english:20,
art:92,
style:"#FFDE89"},
{label:"Barbera",
math:90,
english:55,
art:81,
style:"#E3675C"},
{label:"Belann",
math:50,
english:50,
art:50,
style:"#C2D985"}];

var chartInfo= { y:{min:40, max:100, steps:5,label:"math"},
x:{min:40, max:100, steps:4,label:"english"}
};
*/
//TODO: tooltip for data points

var scatter = function(id, chartInformation, xInput, yInput){
  CHART_PADDING = 30;
  chartInfo = chartInformation;
  x = xInput;
  y = yInput;
  
  //graph outline
  can = document.getElementById(id);
  wid = can.width;
  hei = can.height;
  yData = chartInfo.y;
  ymax = Math.ceil(yData.max);
  ymin = Math.floor(yData.min);
  xData = chartInfo.x;
  xmax = Math.ceil(xData.max);
  xmin = Math.floor(xData.min);
  
  context = can.getContext("2d");
  //context.fillStyle = "#eeeeee";
  context.strokeStyle = "#777";
  //context.fillRect(0,0,wid,hei);

  //chart outlines
  context.font = "14pt italic";
  context.fillStyle = "#777";
  context.moveTo(CHART_PADDING,CHART_PADDING);
  context.lineTo(CHART_PADDING,hei-CHART_PADDING);
  context.lineTo(wid-CHART_PADDING,hei-CHART_PADDING);
  this.fillChart();
  this.plot();
};

scatter.prototype.fillChart = function(){
  
  //for y axis
  var steps = yData.steps;
  var startY = CHART_PADDING;
  var endY = hei - CHART_PADDING;
  var chartHeight = endY-startY;
  var currentY;
  var rangeLength = ymax - ymin;
  var stepSize = rangeLength/steps;
  var yLabel = chartInfo.y.label;
  var yLabelMetrics = context.measureText(yLabel);
  for(var i=0; i<steps; i++){
    currentY = startY + (i/steps)*chartHeight;
    context.moveTo(wid-CHART_PADDING, currentY);
    context.lineTo(CHART_PADDING, currentY);
    //context.fillText(ymin+stepSize*(steps-i), 0, currentY + 4);
  }
  currentY = startY +	chartHeight;
  context.moveTo(CHART_PADDING, currentY );
  context.lineTo(CHART_PADDING,currentY);
  context.save();
  context.translate(CHART_PADDING-10, CHART_PADDING+chartHeight/2+ yLabelMetrics.width/2);
  context.rotate(Math.PI/(-2));
  context.fillText(yLabel,0,0);
  context.restore();
  
  //for x axis
  steps = xData.steps;
  var startX = CHART_PADDING;
  var endX = wid-CHART_PADDING;
  var chartWidth = endX-startX;
  var currentX;
  rangeLength = xmax - xmin;
  stepSize = rangeLength/steps;
  var xLabel = chartInfo.x.label;
  var xLabelMetrics = context.measureText(xLabel);
  context.textAlign = "left";
  for(var i=0; i<steps; i++){
    currentX = startX + (i/steps) *	chartWidth;
    context.moveTo(currentX, startY );
    context.lineTo(currentX,endY);
    //context.fillText(xmin+stepSize*(i), currentX-6, endY+CHART_PADDING/2);
  }
  currentX = startX +	chartWidth;
  context.moveTo(currentX, startY );
  context.lineTo(currentX,endY);
  context.stroke();
  context.fillText(xLabel, CHART_PADDING+(chartWidth/2)-xLabelMetrics.width/2, CHART_PADDING+chartHeight + 15);
};

//create actual dots
scatter.prototype.plot = function(){
  var yDataLabel = chartInfo.y.label;
  var xDataLabel = chartInfo.x.label;
  var yDataRange = ymax - ymin;
  var xDataRange = xmax - xmin;
  var chartHeight = hei- CHART_PADDING*2;
  var chartWidth = wid- CHART_PADDING*2;
  var yPos;
  var xPos;
  context.fillStyle = "blue";
  
  for(var i=0; i<x.length;i++){
    xPos = CHART_PADDING + (x[i][0]-xmin)/
    xDataRange * chartWidth;
    yPos = (hei - CHART_PADDING) -(y[i][0] - ymin)/yDataRange * chartHeight;
    //context.fillRect(xPos-4 ,yPos-4,8,8);
    context.beginPath();
    context.arc(xPos, yPos, 2, 0, 2 * Math.PI);
    context.closePath();
    context.fill();
  }
};

//plot the line
scatter.prototype.plotLine = function(theta){
 
  var yDataRange = ymax - ymin;
  var xDataRange = xmax - xmin;
  var chartHeight = hei- CHART_PADDING*2;
  var chartWidth = wid- CHART_PADDING*2;
  var yPos;
  var xPos;
  context.fillStyle = "red";
  
  for(var i=0; i<x.length;i++){
    xPos = CHART_PADDING + (x[i][0]-xmin)/
    xDataRange * chartWidth;
    yPos = (hei - CHART_PADDING) -(numeric.dot(x[i], theta) - ymin)/yDataRange * chartHeight;
    //context.fillRect(xPos-4 ,yPos-4,8,8);
    context.beginPath();
    context.arc(xPos, yPos, 2, 0, 2 * Math.PI);
    context.closePath();
    context.fill();
  }
};
