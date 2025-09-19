# Separate file for model definition to reduce build memory
def get_model_class():
    """Load PyTorch classes only when needed"""
    import torch
    import torch.nn as nn
    
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
    
    return SimpleCNN
