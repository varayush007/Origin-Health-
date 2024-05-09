import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

class LandmarkDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_size=(256, 256)):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        image = image.resize(self.target_size)

        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'landmarks': landmarks}

csv_file = '/content/drive/MyDrive/role_challenge_dataset_ground_truth.csv'
root_dir = '/content/drive/MyDrive/images'

transform = transforms.Compose([
     transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

landmark_dataset = LandmarkDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

#Splitting the data

from torch.utils.data import DataLoader, random_split
train_size = int(0.6 * len(landmark_dataset))
val_size = int(0.2 * len(landmark_dataset))
test_size = len(landmark_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(landmark_dataset, [train_size, val_size, test_size])


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)