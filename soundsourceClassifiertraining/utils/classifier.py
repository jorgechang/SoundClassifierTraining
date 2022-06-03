"""Model classifier for audio files"""
# Third party modules
import torch.nn as nn


class AudioClassifier(nn.Module):
    """CNN model for audio files"""

    def __init__(self, num_classes):
        """CNN model for audio files"""
        super(AudioClassifier, self).__init__()
        conv_layers = []

        # First Block
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(7, 7), padding="same")
        self.bn1 = nn.BatchNorm2d(8)
        self.gelu1 = nn.GELU()
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_layers += [self.conv1, self.bn1, self.gelu1, self.maxp1]

        # Second Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 5), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.gelu2 = nn.GELU()
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_layers += [self.conv2, self.bn2, self.gelu2, self.maxp2]

        # Third Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding="same")
        self.bn3 = nn.BatchNorm2d(32)
        self.gelu3 = nn.GELU()
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_layers += [self.conv3, self.bn3, self.gelu3, self.maxp3]

        # Fourth Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same")
        self.bn4 = nn.BatchNorm2d(64)
        self.gelu4 = nn.GELU()
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_layers += [self.conv4, self.bn4, self.gelu4, self.maxp4]

        self.flatten = nn.Flatten()  # Flatten for input to linear layer
        self.fc1 = nn.Linear(5376, 128)  # Linear layer
        self.dp = nn.Dropout(p=0.3)  # Dropout to avoid overfitting
        self.fc2 = nn.Linear(128, num_classes)  # Classification layer

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        """Forward propagation"""
        # Run the convolutional blocks
        x = self.conv(x)

        # Flatten for input to linear layer
        x = x.view(x.shape[0], -1)

        # Run the linear layers
        x = self.dp(x)
        x = self.fc1(x)
        x = self.dp(x)
        x = self.fc2(x)

        # Final output
        return x
