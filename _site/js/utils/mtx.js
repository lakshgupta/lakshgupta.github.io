var Matrix = function(rows, cols, data) {
	this.rows = rows;
	this.cols = cols;
	this.length = rows * cols;
	if (!data) {
		this.data = null;
	} else {
		this.data = data;
	}
};


		
