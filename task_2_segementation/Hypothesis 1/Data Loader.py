import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # Ensure masks are grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}


# transformation of the data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image_dir = '/content/drive/MyDrive/images'
mask_dir = '/content/drive/MyDrive/masks'

segmentation_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

# splitting the data

train_size_seg = int(0.6 * len(segmentation_dataset))
val_size_seg = int(0.2 * len(segmentation_dataset))
test_size_seg = len(segmentation_dataset) - train_size_seg - val_size_seg

train_dataset_seg, val_dataset_seg, test_dataset_seg = random_split(segmentation_dataset,
                                                                    [train_size_seg, val_size_seg, test_size_seg])

# Define batch size and create data loaders
batch_size_seg = 8
train_loader_seg = DataLoader(train_dataset_seg, batch_size=batch_size_seg, shuffle=True)
val_loader_seg = DataLoader(val_dataset_seg, batch_size=batch_size_seg, shuffle=False)
test_loader_seg = DataLoader(test_dataset_seg, batch_size=batch_size_seg, shuffle=False)