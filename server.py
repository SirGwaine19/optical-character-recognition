from flask import Flask, request, jsonify, render_template
import sys
import os

# Add the current directory to Python path to ensure ocr module is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ocr import OCRModel
    print("✅ OCR Model imported successfully", flush=True)
except ImportError as e:
    print(f"❌ Failed to import OCR Model: {e}", flush=True)
    sys.exit(1)

app = Flask(__name__)

# Initialize OCR model with proper error handling
try:
    print("🚀 Initializing OCR Model...", flush=True)
    ocr_model = OCRModel()
    print("✅ OCR Model initialized successfully", flush=True)
except Exception as e:
    print(f"❌ Failed to initialize OCR Model: {e}", flush=True)
    sys.exit(1)

@app.route('/')
def index():
    return render_template('ocr.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        print("📨 Received OCR request", flush=True)
        data = request.get_json()
        
        if not data:
            print("❌ No JSON data received", flush=True)
            return jsonify({"error": "No data provided"}), 400
        
        if 'train' in data:
            print("🎯 Training request detected", flush=True)
            
            # Extract training data
            if 'trainArray' not in data:
                print("❌ No trainArray found in request", flush=True)
                return jsonify({"error": "No training data provided"}), 400
            
            train_array = data['trainArray']
            print(f"📊 Received {len(train_array)} training samples", flush=True)
            
            # Extract features and labels
            try:
                X_train = []
                y_train = []
                
                for i, item in enumerate(train_array):
                    if 'y0' not in item or 'label' not in item:
                        print(f"⚠️ Sample {i} missing 'y0' or 'label' field", flush=True)
                        continue
                    
                    X_train.append(item['y0'])
                    y_train.append(item['label'])
                
                if not X_train or not y_train:
                    print("❌ No valid training samples found", flush=True)
                    return jsonify({"error": "No valid training samples"}), 400
                
                print(f"✅ Processed {len(X_train)} valid training samples", flush=True)
                print(f"📋 Labels: {y_train}", flush=True)
                
            except Exception as e:
                print(f"❌ Error processing training data: {str(e)}", flush=True)
                return jsonify({"error": f"Data processing error: {str(e)}"}), 400

            # Train the model
            try:
                print("🔄 Starting model training...", flush=True)
                sys.stdout.flush()
                
                loss, acc = ocr_model.train(X_train, y_train)
                
                print(f"🎉 Training completed! Loss: {loss}, Accuracy: {acc}", flush=True)
                
                response = {
                    "status": "training done",
                    "loss": float(loss),
                    "accuracy": float(acc),
                    "samples_trained": len(X_train)
                }
                
                print(f"📤 Sending response: {response}", flush=True)
                return jsonify(response)
                
            except Exception as e:
                print(f"❌ Training error: {str(e)}", flush=True)
                return jsonify({"error": f"Training failed: {str(e)}"}), 500
        
        elif 'predict' in data:
            print("🔮 Prediction request detected", flush=True)
            
            if 'image' not in data:
                return jsonify({"error": "No image data provided"}), 400
            
            try:
                prediction = ocr_model.predict(data['image'])
                print(f"🎯 Prediction result: {prediction}", flush=True)
                
                return jsonify({
                    "status": "prediction done",
                    "prediction": int(prediction)
                })
                
            except Exception as e:
                print(f"❌ Prediction error: {str(e)}", flush=True)
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        else:
            print("❓ Unknown request type", flush=True)
            return jsonify({"error": "Unknown request type"}), 400
    
    except Exception as e:
        print(f"❌ General error in OCR endpoint: {str(e)}", flush=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/status')
def status():
    """Health check endpoint"""
    return jsonify({
        "status": "OCR service is running",
        "model_loaded": ocr_model.model is not None
    })

if __name__ == '__main__':
    print("🌐 Starting Flask OCR application...", flush=True)
    print("📍 Navigate to http://127.0.0.1:5000 to access the OCR demo", flush=True)
    
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=True,
        threaded=True
    )
