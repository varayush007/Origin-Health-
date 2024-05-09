# Defining the UNet architecture for segmentation
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Defining encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Defining decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # Additional upsampling layer
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Defining Binary Cross-Entropy Loss for segmentation
criterion_seg = nn.BCEWithLogitsLoss()

# Defining a new segmentation model instance
segmentation_model_h2 = UNet()

# Defining new optimizer with adjusted learning rate
optimizer_seg_h2 = torch.optim.Adam(segmentation_model_h2.parameters(), lr=0.0001)