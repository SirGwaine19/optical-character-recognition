import json
import math
import numpy as np  
import os
import random

NUM_DIGITS = 10

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    NN_FILE_PATH = 'nn.json'
    EPOCHS_PER_BATCH = 3  # Train multiple times on each batch for better learning

    def __init__(self, num_hidden_nodes, use_file=True):
        self._use_file = use_file
        self.num_hidden_nodes = num_hidden_nodes
        
        
        if not self._use_file or not os.path.exists(self.NN_FILE_PATH):
            self._rand_initialize_weights(400, num_hidden_nodes)
        else:
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        """Initialize weights using Xavier initialization for better convergence"""
        
        epsilon1 = np.sqrt(6.0 / (size_in + size_out))
        self.theta1 = np.random.uniform(-epsilon1, epsilon1, (size_out, size_in))
        
        epsilon2 = np.sqrt(6.0 / (size_out + NUM_DIGITS))
        self.theta2 = np.random.uniform(-epsilon2, epsilon2, (NUM_DIGITS, size_out))
        
        # Initialize biases to small random values
        self.input_layer_bias = np.random.uniform(-0.1, 0.1, (1, size_out))
        self.hidden_layer_bias = np.random.uniform(-0.1, 0.1, (1, NUM_DIGITS))

    def sigmoid(self, z):
        """Vectorized sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow

    def sigmoid_prime(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def _preprocess_image(self, image_data):
        """Preprocess image: normalize and center"""
        img = np.array(image_data).reshape(20, 20).astype(float)
        
        # Find bounding box of the digit
        rows = np.any(img > 0, axis=1)
        cols = np.any(img > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Empty image, return normalized zeros
            return np.zeros(400)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the digit region
        digit_region = img[rmin:rmax+1, cmin:cmax+1]
        
        # Center the digit in a 20x20 canvas
        h, w = digit_region.shape
        offset_y = (20 - h) // 2
        offset_x = (20 - w) // 2
        
        centered = np.zeros((20, 20))
        centered[offset_y:offset_y+h, offset_x:offset_x+w] = digit_region
        
        # Normalize to [0, 1]
        if np.max(centered) > 0:
            centered = centered / np.max(centered)
        
        return centered.flatten()

    def train(self, train_array):
        """Train the network using backpropagation with multiple epochs"""
        
        for epoch in range(self.EPOCHS_PER_BATCH):
            
            shuffled = list(train_array)
            random.shuffle(shuffled)
            
            for data in shuffled:
                # Preprocess the image
                y0 = self._preprocess_image(data['y0']).reshape(400, 1)
                
                # Forward propagation
                y1 = np.dot(self.theta1, y0)
                sum1 = y1 + self.input_layer_bias.T
                y1 = self.sigmoid(sum1)

                # Output layer
                y2 = np.dot(self.theta2, y1)
                y2 = y2 + self.hidden_layer_bias.T
                y2 = self.sigmoid(y2)

                # Backpropagation
                actual_vals = np.zeros((NUM_DIGITS, 1))
                actual_vals[data['label']] = 1
                
                output_errors = actual_vals - y2
                hidden_errors = np.multiply(
                    np.dot(self.theta2.T, output_errors),
                    self.sigmoid_prime(sum1)
                )

                # Weight updates
                self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, y0.T)
                self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.T)
                self.hidden_layer_bias += self.LEARNING_RATE * output_errors.T
                self.input_layer_bias += self.LEARNING_RATE * hidden_errors.T

    def predict(self, test):
        """Predict the digit from input data and return prediction with confidence"""
        # Preprocess the image
        y0 = self._preprocess_image(test).reshape(400, 1)
        
        # Forward propagation
        y1 = np.dot(self.theta1, y0)
        y1 = y1 + self.input_layer_bias.T
        y1 = self.sigmoid(y1)

        y2 = np.dot(self.theta2, y1)
        y2 = y2 + self.hidden_layer_bias.T
        y2 = self.sigmoid(y2)

        results = y2.flatten().tolist()
        predicted_digit = results.index(max(results))
        confidence = max(results)
        
        # Return both digit and confidence
        return {"digit": predicted_digit, "confidence": float(confidence), "all_scores": results}

    def save(self):
        """Save the neural network weights to a JSON file"""
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": [row.tolist() for row in self.theta1],
            "theta2": [row.tolist() for row in self.theta2],
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        
        with open(self.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        """Load the neural network weights from a JSON file"""
        if not self._use_file:
            return

        with open(self.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        
        self.theta1 = np.array(nn['theta1'])
        self.theta2 = np.array(nn['theta2'])
        self.input_layer_bias = np.array(nn['b1'])
        self.hidden_layer_bias = np.array(nn['b2'])
