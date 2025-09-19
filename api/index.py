# Minimal imports to reduce build memory
from flask import Flask, jsonify, request
import os
import gc

# --- 1. Model definition is in separate file to reduce build memory ---

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- 3. Model Loading (Ultra Lazy Loading) ---
# Global variables for models (will be loaded on first use)
model1 = None
model2 = None
model_path = os.path.join(os.path.dirname(__file__), '..', 'model')

def load_model(model_id):
    """Ultra lazy load models to minimize build memory usage"""
    global model1, model2
    
    # Import PyTorch only when needed
    import torch
    import torch.nn as nn
    from PIL import Image
    import torchvision.transforms as transforms
    
    device = torch.device("cpu")
    
    if model_id == "1" and model1 is None:
        print("Loading model 1...")
        from .model_def import get_model_class
        SimpleCNN = get_model_class()
        model1 = SimpleCNN().to(device)
        try:
            model1.load_state_dict(torch.load(
                os.path.join(model_path, 'best_cnn_model.pth'), 
                map_location=device,
                weights_only=True  # Safer loading
            ))
            model1.eval()
            print("Model 1 loaded successfully")
            # Force garbage collection to free up memory
            gc.collect()
        except FileNotFoundError as e:
            print(f"Error loading model 1: {e}")
            return None
    elif model_id == "2" and model2 is None:
        print("Loading model 2...")
        from .model_def import get_model_class
        SimpleCNN = get_model_class()
        model2 = SimpleCNN().to(device)
        try:
            model2.load_state_dict(torch.load(
                os.path.join(model_path, 'best_cnn_model.pth'), 
                map_location=device,
                weights_only=True  # Safer loading
            ))
            model2.eval()
            print("Model 2 loaded successfully")
            # Force garbage collection to free up memory
            gc.collect()
        except FileNotFoundError as e:
            print(f"Error loading model 2: {e}")
            return None
    
    return model1 if model_id == "1" else model2

# --- 4. Image Preprocessing (loaded on demand) ---
def get_preprocess():
    """Get preprocessing transforms only when needed"""
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Convert to grayscale
        transforms.Resize((144, 144)),               # Resize the image to match training size
        transforms.ToTensor(),                       # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])

# --- 5. Set up CORS Headers ---
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, PUT, DELETE'
    return response

app.after_request(add_cors_headers)

# --- 6. Define the API Endpoint ---
@app.route('/api/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Import required modules only when needed
        import torch
        from PIL import Image
        
        image_file = request.files['image']
        model_n = request.form.get('model', '1')  # Default to model 1 if not specified
        
        # Open the image file
        image = Image.open(image_file.stream).convert('RGB')
        
        # Get preprocessing transforms
        preprocess = get_preprocess()
        device = torch.device("cpu")
        
        # Preprocess the image and add a batch dimension
        img_tensor = preprocess(image)
        img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension and send to device

        # Load the model (lazy loading)
        selected_model = load_model(model_n)
        if selected_model is None:
            return jsonify({"error": "Failed to load model"}), 500
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            output = selected_model(img_tensor)
            # Apply sigmoid to get a probability and extract the value
            probability = torch.sigmoid(output).item()
        
        # Clear GPU cache and force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Probability: {probability}")
        
        # Determine the result string
        if probability < 0.5: # Using a standard 0.5 threshold for binary classification
            result = f"Person is not affected with Pneumonia. (Model Version: {model_n}) Probability: {probability:.4f}"
        else:
            result = f"Person is affected with Pneumonia. (Model Version: {model_n}) Probability: {probability:.4f}"

        return jsonify({"text": result})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process image."}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Pneumonia detection API is running"})

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Pneumonia Detection API", "endpoints": ["/api/process_image", "/api/health"]})

# This is the entry point for Vercel
def handler(request):
    return app(request.environ, lambda *args: None)
