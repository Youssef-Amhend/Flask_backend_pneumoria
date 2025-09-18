import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image
import torchvision.transforms as transforms
import io
import os

# --- 1. Define your PyTorch Model Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Based on the actual saved model architecture
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # Input size: 128 channels * 18x18 (after 3 max pools from 144x144) = 41472
            nn.Dropout(0.5),  # fc.0
            nn.Linear(128 * 18 * 18, 256),  # fc.1
            nn.ReLU(),  # fc.2
            nn.Dropout(0.5),  # fc.3
            nn.Linear(256, 1)  # fc.4 - Output is a single value for binary classification
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1) # Flatten the feature maps
        x = self.fc(x)
        return x

# --- 2. Initialize the Flask App ---
app = Flask(__name__)

# --- 3. Load the PyTorch Models ---
# Use CPU for Vercel deployment (no GPU available)
device = torch.device("cpu")

# Instantiate the model architecture
model1 = SimpleCNN().to(device)
model2 = SimpleCNN().to(device)

# Load the learned weights from the .pth files
# Use absolute paths for Vercel deployment
model_path = os.path.join(os.path.dirname(__file__), '..', 'model')
try:
    model1.load_state_dict(torch.load(os.path.join(model_path, 'best_cnn_model.pth'), map_location=device))
    model2.load_state_dict(torch.load(os.path.join(model_path, 'best_cnn_model.pth'), map_location=device))
    print("Models loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure model files are in the 'model' directory.")
    
# Set models to evaluation mode
model1.eval()
model2.eval()

# --- 4. Define Image Preprocessing ---
preprocess = transforms.Compose([
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
        image_file = request.files['image']
        model_n = request.form.get('model', '1')  # Default to model 1 if not specified
        
        # Open the image file
        image = Image.open(image_file.stream).convert('RGB')
        
        # Preprocess the image and add a batch dimension
        img_tensor = preprocess(image)
        img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension and send to device

        # Select the model and perform prediction
        selected_model = model1 if model_n == "1" else model2
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            output = selected_model(img_tensor)
            # Apply sigmoid to get a probability and extract the value
            probability = torch.sigmoid(output).item()
        
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
