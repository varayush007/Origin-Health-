# Defining the UNet architecture for segmentation with ResNet-50 backbone
class UNetWithResNet(nn.Module):
    def __init__(self):
        super(UNetWithResNet, self).__init__()
        # Loading pre-trained ResNet-50 as encoder
        self.encoder = models.resnet50(pretrained=True)
        # Modifing the first layer to accept 3 channels instead of 1
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Defining decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # Forward pass through ResNet-50 encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        # Forward pass through decoder
        x = self.decoder(x)
        return x


# Defining Binary Cross-Entropy Loss for segmentation
criterion_seg = nn.BCEWithLogitsLoss()

# Defining a new segmentation model instance with ResNet-50 backbone
segmentation_model_h3 = UNetWithResNet()

# Freezing the ResNet-50 layers
for param in segmentation_model_h3.encoder.parameters():
    param.requires_grad = False

# Defining new optimizer with adjusted learning rate
optimizer_seg_h3 = torch.optim.Adam(segmentation_model_h3.decoder.parameters(), lr=0.0001)