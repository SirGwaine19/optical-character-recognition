var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10,
    data: new Array(400).fill(0),
    trainArray: [],
    trainingRequestCount: 0,
    BATCH_SIZE: 5,
    HOST: window.location.origin,
    PORT: '',

    onLoadFunction: function() {
        var canvas = document.getElementById("canvas");
        if (!canvas) {
            console.error("Canvas element not found!");
            return;
        }
        
        // Set canvas dimensions
        canvas.width = this.CANVAS_WIDTH;
        canvas.height = this.CANVAS_WIDTH;
        canvas.isDrawing = false;
        
        var ctx = canvas.getContext("2d");
        if (!ctx) {
            console.error("Could not get canvas context!");
            return;
        }

        // Add event listeners
        canvas.addEventListener("mousedown", e => this.onMouseDown(e, ctx, canvas));
        canvas.addEventListener("mousemove", e => this.onMouseMove(e, ctx, canvas));
        canvas.addEventListener("mouseup", e => this.onMouseUp(e));
        
        // Add touch events for mobile support
        canvas.addEventListener("touchstart", e => {
            e.preventDefault();
            var touch = e.touches[0];
            var rect = canvas.getBoundingClientRect();
            var mouseEvent = new MouseEvent("mousedown", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener("touchmove", e => {
            e.preventDefault();
            var touch = e.touches[0];
            var rect = canvas.getBoundingClientRect();
            var mouseEvent = new MouseEvent("mousemove", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener("touchend", e => {
            e.preventDefault();
            var mouseEvent = new MouseEvent("mouseup", {});
            canvas.dispatchEvent(mouseEvent);
        });

        this.drawGrid(ctx);
        console.log("OCR Demo initialized successfully");
    },

    drawGrid: function(ctx) {
        ctx.strokeStyle = "#0000FF";
        ctx.lineWidth = 1;
        
        // Draw vertical lines
        for (var x = this.PIXEL_WIDTH; x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();
        }
        
        // Draw horizontal lines
        for (var y = this.PIXEL_WIDTH; y < this.CANVAS_WIDTH; y += this.PIXEL_WIDTH) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        this.fillSquare(ctx, x, y);
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) return;
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        this.fillSquare(ctx, x, y);
    },

    onMouseUp: function(e) {
        var canvas = document.getElementById("canvas");
        if (canvas) {
            canvas.isDrawing = false;
        }
    },

    fillSquare: function(ctx, x, y) {
        // Ensure coordinates are within bounds
        if (x < 0 || y < 0 || x >= this.CANVAS_WIDTH || y >= this.CANVAS_WIDTH) {
            return;
        }
        
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        
        // Ensure pixel coordinates are within bounds
        if (xPixel < 0 || yPixel < 0 || xPixel >= this.TRANSLATED_WIDTH || yPixel >= this.TRANSLATED_WIDTH) {
            return;
        }
        
        var dataIndex = yPixel * this.TRANSLATED_WIDTH + xPixel;
        
        // Only fill if not already filled
        if (this.data[dataIndex] === 0) {
            this.data[dataIndex] = 1;
            ctx.fillStyle = "#FFFFFF";
            ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
                        this.PIXEL_WIDTH, this.PIXEL_WIDTH);
        }
    },

    train: function() {
        var digitInput = document.getElementById("digit");
        if (!digitInput) {
            alert("Digit input field not found!");
            return;
        }
        
        var digitVal = digitInput.value.trim();
        
        // Validate input
        if (!digitVal) {
            alert("Please enter a digit (0-9) to train");
            return;
        }
        
        var digitNum = parseInt(digitVal);
        if (isNaN(digitNum) || digitNum < 0 || digitNum > 9) {
            alert("Please enter a valid digit between 0 and 9");
            return;
        }
        
        // Check if something is drawn
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit first");
            return;
        }

        // Add to training array
        this.trainArray.push({
            "y0": this.data.slice(), // Create a copy of the data
            "label": digitNum
        });
        
        this.trainingRequestCount++;
        
        console.log(`Added training sample: digit ${digitNum}, total samples: ${this.trainArray.length}`);
        
        // Send data when batch is full
        if (this.trainingRequestCount >= this.BATCH_SIZE) {
            console.log(`Sending batch of ${this.trainArray.length} samples for training`);
            this.sendData({ 
                trainArray: this.trainArray.slice(), // Send a copy
                train: true 
            });
            this.trainArray = []; // Clear the array
            this.trainingRequestCount = 0;
        } else {
            alert(`Training sample added! (${this.trainingRequestCount}/${this.BATCH_SIZE} in current batch)`);
        }
        
        // Clear canvas after adding sample
        this.resetCanvas();
        digitInput.value = "";
    },

    test: function() {
        // Check if something is drawn
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit to test");
            return;
        }

        console.log("Sending prediction request");
        this.sendData({ 
            image: this.data.slice(),
            predict: true 
        });
    },

    sendData: function(json) {
        console.log("Sending data:", json);
        
        var url = this.HOST + "/ocr";
        console.log("Request URL:", url);
        
        // Show loading message
        if (json.train) {
            console.log("Training request sent...");
        } else if (json.predict) {
            console.log("Prediction request sent...");
        }
        
        fetch(url, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(json)
        })
        .then(response => {
            console.log("Response status:", response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Response data:", data);
            
            // Handle different response types
            if (data.prediction !== undefined) {
                alert(`Predicted digit: ${data.prediction}`);
            } else if (data.status === "training done") {
                alert(`Training completed!\nLoss: ${data.loss.toFixed(4)}\nAccuracy: ${data.accuracy.toFixed(4)}\nSamples: ${data.samples_trained}`);
            } else if (data.status) {
                alert(data.status);
            } else if (data.error) {
                alert("Error: " + data.error);
            } else {
                console.log("Unexpected response format:", data);
                alert("Operation completed (see console for details)");
            }
        })
        .catch(err => {
            console.error("Request failed:", err);
            alert("Error: " + err.message + "\nCheck console for details");
        });
    },

    resetCanvas: function() {
        var canvas = document.getElementById("canvas");
        if (!canvas) {
            console.error("Canvas element not found!");
            return;
        }
        
        var ctx = canvas.getContext("2d");
        if (!ctx) {
            console.error("Could not get canvas context!");
            return;
        }
        
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Reset data array
        this.data.fill(0);
        
        // Redraw grid
        this.drawGrid(ctx);
        
        console.log("Canvas reset");
    },

    // Additional utility functions
    forceTrainBatch: function() {
        if (this.trainArray.length > 0) {
            console.log(`Force sending batch of ${this.trainArray.length} samples`);
            this.sendData({ 
                trainArray: this.trainArray.slice(),
                train: true 
            });
            this.trainArray = [];
            this.trainingRequestCount = 0;
        } else {
            alert("No training samples to send");
        }
    },

    clearTrainingData: function() {
        this.trainArray = [];
        this.trainingRequestCount = 0;
        alert("Training data cleared");
    },

    showCurrentData: function() {
        console.log("Current canvas data:", this.data);
        console.log("Training array length:", this.trainArray.length);
        console.log("Training request count:", this.trainingRequestCount);
    }
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded, initializing OCR Demo");
    ocrDemo.onLoadFunction();
});

// Fallback for older browsers
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        ocrDemo.onLoadFunction();
    });
} else {
    ocrDemo.onLoadFunction();
}