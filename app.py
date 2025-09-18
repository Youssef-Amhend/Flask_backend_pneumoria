import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image
import torchvision.transforms as transforms
import io

# --- 1. Define your PyTorch Model Architecture ---
# IMPORTANT: This class MUST exactly match the architecture of the model
# you saved in your .pth files. This is a placeholder example.
# Replace it with your actual model definition.
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
# Use a device that's available (CUDA if you have a GPU, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model architecture
model1 = SimpleCNN().to(device)
model2 = SimpleCNN().to(device)

# Load the learned weights from the .pth files
# Update these paths to where your model files are located.
try:
    model1.load_state_dict(torch.load('./model/best_cnn_model.pth', map_location=device))
    model2.load_state_dict(torch.load('./model/best_cnn_model.pth', map_location=device))
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure 'pneumonia_detection_model_v1.pth' and 'pneumonia_detection_model_v2.pth' are in a 'model' subfolder.")
    # Exit or handle as needed if models are essential
    
# Set models to evaluation mode (important for layers like dropout and batch norm)
model1.eval()
model2.eval()

# --- 4. Define Image Preprocessing ---
# This transformation pipeline should match what you used for training your PyTorch model
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Convert to grayscale
    transforms.Resize((144, 144)),               # Resize the image to match training size
    transforms.ToTensor(),                       # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust mean/std if you used different values)
])


# --- 5. Set up CORS Headers ---
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, PUT, DELETE'
    return response

app.after_request(add_cors_headers)


# --- 6. Define the API Endpoint ---
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_file = request.files['image']
        model_n = request.form['model']
        
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
        print(probability)
        # Determine the result string
        if probability < 0.5: # Using a standard 0.5 threshold for binary classification
            s = f"Person is not affected with Pneumonia. (Model Version: {model_n}) {probability: {probability:.4f}}"
        else:
            s = f"Person is affected with Pneumonia. (Model Version: {model_n}) {probability: {probability:.4f}}"

        return jsonify({"text": s})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process image."}), 500


# --- 7. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)