function neuron(context, x, y, radius, label) {
    this.context = context;
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.label = label;
    this.draw();
}

neuron.prototype.draw = function() {
    this.context.circle(this.x, this.y, this.radius, defaultLine, 1, "white");
    if (typeof this.label !== "undefined") {
	    this.context.mathText(this.label, this.x, this.y+4, {"text-align": "center"});
    }
};

neuron.prototype.connectTo = function(output, color) {
  // Connects the neuron to output.  output may be either a single
  // neuron or an array containing multiple neurons.
  if (! Array.isArray(output)) {
	  output = [output];
  }
  color = selfOrDefault(color, defaultLine);
  output.forEach(function(outputNeuron, index, array) {
	  var d = {}, len, n = {};
	  d.x = outputNeuron.x-this.x;
	  d.y = outputNeuron.y-this.y;
	  len = Math.sqrt(d.x*d.x+d.y*d.y);
	  n.x = d.x/len;
	  n.y = d.y/len;
	  var xStart = this.x+n.x*this.radius;
	  var yStart = this.y+n.y*this.radius;
	  var xEnd = outputNeuron.x-n.x*this.radius;
	  var yEnd = outputNeuron.y-n.y*this.radius;
	  this.context.arrow(xStart, yStart, xEnd, yEnd, color, 0.7);
  }, this);
};

neuron.prototype.input = function(label) {
    var xStart = this.x-this.radius-25;
    var xEnd = this.x-this.radius;
    this.context.arrow(xStart, this.y, xEnd, this.y);
    if (typeof label !== "undefined") {
	var width = this.context.measureText(label).width;
	this.context.mathText(label, xStart-width-10, this.y+5);
    }
};

neuron.prototype.output = function(label) {
    var xStart = this.x+this.radius;
    var xEnd = this.x+this.radius+25;
    this.context.arrow(xStart, this.y, xEnd, this.y, defaultLine, 0.7);
    if (typeof label !== "undefined") {
	this.context.mathText(label, xEnd+10, this.y+5);
    }
};    

function connectLayers(layerIn, layerOut, color) {
    // Draw arrows connecting all the neurons in layerIn to all the
    // neurons in layerOut
    color = selfOrDefault(color, defaultLine);
    layerIn.forEach(function(n) {
	    n.connectTo(layerOut, color);
    });
}
