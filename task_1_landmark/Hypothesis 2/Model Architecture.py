import torch
import torch.nn as nn

class LandmarkDetectionModel(nn.Module):
    def __init__(self):
        super(LandmarkDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1_input_size = 512 * 8 * 8
        self.fc1 = nn.Linear(self.fc1_input_size, 1000)
        self.fc2 = nn.Linear(1000, 8)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        print("After conv1:", x.size())
        x = self.pool(x)
        print("After pooling 1:", x.size())
        x = torch.relu(self.bn2(self.conv2(x)))
        print("After conv2:", x.size())
        x = self.pool(x)
        print("After pooling 2:", x.size())
        x = torch.relu(self.bn3(self.conv3(x)))
        print("After conv3:", x.size())
        x = self.pool(x)
        print("After pooling 3:", x.size())
        x = torch.relu(self.bn4(self.conv4(x)))
        print("After conv4:", x.size())
        x = self.pool(x)
        print("After pooling 4:", x.size())
        x = x.view(-1, self.fc1_input_size)
        print("After flattening:", x.size())
        x = torch.relu(self.fc1(x))
        print("After fc1:", x.size())
        x = self.dropout(x)
        x = self.fc2(x)
        print("Final output size:", x.size())
        return x


model = LandmarkDetectionModel()

#Loss
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)