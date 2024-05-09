import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# Defining the Segmentation Dataset class
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
        mask = Image.open(mask_name).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}

# Defining the augmented transformations
aug_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()
])

# Defining image and mask directories
image_dir = '/content/drive/MyDrive/images'
mask_dir = '/content/drive/MyDrive/masks'

# Creating augmented dataset
augmented_dataset = SegmentationDataset(image_dir, mask_dir, transform=aug_transform)


# Spliting the augmented dataset
train_size_aug = int(0.6 * len(augmented_dataset))
val_size_aug = int(0.2 * len(augmented_dataset))
test_size_aug = len(augmented_dataset) - train_size_aug - val_size_aug

train_dataset_aug, val_dataset_aug, test_dataset_aug = random_split(augmented_dataset,
                                                                    [train_size_aug, val_size_aug, test_size_aug])

# Defining batch size and create data loaders for augmented dataset
batch_size_aug = 16
train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size_aug, shuffle=True)
val_loader_aug = DataLoader(val_dataset_aug, batch_size=batch_size_aug, shuffle=False)
test_loader_aug = DataLoader(test_dataset_aug, batch_size=batch_size_aug, shuffle=False)