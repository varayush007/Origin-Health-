# Define the U-Net architecture for segmentation
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define encoder layers
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
        # Define decoder layers
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
        print("After encoder:", x.size())  # Print size after encoder
        x = self.decoder(x)
        print("After decoder:", x.size())  # Print size after decoder
        return x


#loss
criterion_seg = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for segmentation
optimizer_seg = torch.optim.Adam(segmentation_model.parameters(), lr=0.001)