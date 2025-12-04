import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from ocr import OCRNeuralNetwork
import numpy as np  # type: ignore

# Initialize the OCR neural network
nn = OCRNeuralNetwork(15, True)  

class OCRServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve static files (HTML, CSS, JS)"""
        # Get the requested path
        path = self.path.strip('/')
        if not path or path == '/':
            path = 'ocr.html'
        
        # Map paths to files
        file_map = {
            'ocr.html': 'ocr.html',
            'ocr.js': 'ocr.js',
            'ocr.css': 'ocr.css',
        }
        
        # Determine content type
        content_types = {
            'html': 'text/html',
            'js': 'application/javascript',
            'css': 'text/css',
        }
        
        if path in file_map:
            filename = file_map[path]
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine content type from file extension
                ext = filename.split('.')[-1]
                content_type = content_types.get(ext, 'text/plain')
                
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except FileNotFoundError:
                self.send_response(404)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"File not found")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error: {str(e)}".encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not found")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length', 0))
        content = self.rfile.read(var_len)
        payload = json.loads(content.decode('utf-8'))

        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()
        elif payload.get('predict'):
            try:
                prediction_result = nn.predict(payload['image'])
                response = {
                    "type": "test", 
                    "result": prediction_result['digit'],
                    "confidence": prediction_result['confidence'],
                    "all_scores": prediction_result['all_scores']
                }
            except Exception as e:
                print(f"Error during prediction: {e}")
                import traceback
                traceback.print_exc()
                response_code = 500
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))
        return

    def log_message(self, format, *args):
        return

if __name__ == '__main__':
    server = HTTPServer(('localhost', 5000), OCRServerHandler)
    print("Server started on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
