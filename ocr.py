import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import os
import sys

MODEL_PATH = 'model/ocr_model.h5'

class OCRModel:
    def __init__(self):
        self.model = None
        if not os.path.exists('model'):
            os.makedirs('model')
        self.build_model()
        self.load_model()

    def build_model(self):
        self.model = Sequential([
            Flatten(input_shape=(20, 20)),  # 20x20 pixel input
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')  # Output digits 0-9
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10):
        print(f"➡️ Training on {len(X_train)} samples", flush=True)
        sys.stdout.flush()  # Force flush output
        
        # Convert to numpy arrays and ensure proper format
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        
        # Normalize pixel values to 0-1 range if they're in 0-255 range
        if X_train.max() > 1.0:
            X_train = X_train / 255.0
        
        # Reshape if needed (ensure proper dimensions)
        if len(X_train.shape) == 2:
            # If X_train is flattened, reshape to (samples, 20, 20)
            X_train = X_train.reshape(-1, 20, 20)
        elif len(X_train.shape) == 1:
            # If single sample, reshape appropriately
            X_train = X_train.reshape(1, 20, 20)
        
        print(f"📊 Data shape: {X_train.shape}, Labels shape: {y_train.shape}", flush=True)
        print(f"🔄 Starting training for {epochs} epoch(s)...", flush=True)
        sys.stdout.flush()

        try:
            history = self.model.fit(
                X_train, y_train, 
                epochs=epochs, 
                verbose=1,  # Show training progress
                batch_size=32
            )

            loss = history.history['loss'][-1]  # Get last epoch's loss
            acc = history.history.get('accuracy', [None])[-1]  # Get last epoch's accuracy

            print(f"✅ Training complete! Loss: {loss:.4f}, Accuracy: {acc:.4f if acc else 'N/A'}", flush=True)
            sys.stdout.flush()

            # Save the model after training
            self.save_model()
            print("💾 Model saved successfully!", flush=True)

            return float(loss), float(acc) if acc else 0.0

        except Exception as e:
            print(f"❌ Training failed: {str(e)}", flush=True)
            sys.stdout.flush()
            return 0.0, 0.0

    def predict(self, X):
        try:
            # Ensure proper format
            X = np.array(X, dtype=np.float32)
            
            # Normalize if needed
            if X.max() > 1.0:
                X = X / 255.0
            
            # Reshape if needed
            if len(X.shape) == 1:
                X = X.reshape(1, 20, 20)
            elif len(X.shape) == 2 and X.shape != (20, 20):
                X = X.reshape(20, 20)
                X = X.reshape(1, 20, 20)
            
            pred = self.model.predict(X, verbose=0)
            predicted_digit = int(np.argmax(pred[0]))
            confidence = float(np.max(pred[0]))
            
            print(f"🔮 Prediction: {predicted_digit} (confidence: {confidence:.3f})", flush=True)
            
            return predicted_digit
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}", flush=True)
            return 0

    def save_model(self):
        try:
            self.model.save(MODEL_PATH)
            print(f"💾 Model saved to {MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}", flush=True)

    def load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(f"📂 Model loaded from {MODEL_PATH}", flush=True)
            else:
                print("📝 No existing model found. Will train from scratch.", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to load model: {str(e)}. Building new model.", flush=True)
            self.build_model()
