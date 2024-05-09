import torch
import torch.nn as nn

class LandmarkDetectionModel(nn.Module):
    def __init__(self):
        super(LandmarkDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc1_input_size = 64 * 64 * 64
        self.fc1 = nn.Linear(self.fc1_input_size, 100)
        self.fc2 = nn.Linear(100, 8)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print("After conv1:", x.size())
        x = self.pool(x)
        print("After max pooling 1:", x.size())
        x = torch.relu(self.conv2(x))
        print("After conv2:", x.size())
        x = self.pool(x)
        print("After max pooling 2:", x.size())
        x = x.view(-1, self.fc1_input_size)
        print("After flattening:", x.size())
        x = torch.relu(self.fc1(x))
        print("After fc1:", x.size())
        x = self.fc2(x)
        print("Final output size:", x.size())
        return x

model = LandmarkDetectionModel()

#Loss

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)