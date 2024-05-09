# Defining the ResNet-based landmark detection model
class LandmarkDetectionModel(nn.Module):
    def __init__(self, num_classes=8):
        super(LandmarkDetectionModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the last layer (the fully connected layer)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        # Add custom fully connected layers for landmark detection
        self.fc1 = nn.Linear(resnet.fc.in_features, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        features = self.resnet_features(x)
        features = features.view(features.size(0), -1)  # Flatten the feature map
        x = torch.relu(self.fc1(features))
        x = self.fc2(x)
        return x




model = LandmarkDetectionModel()