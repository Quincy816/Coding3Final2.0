import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 8：2划分
valid_split = 0.2
seed = 42
batch_size = 64
root_dir = r'data_set/flower_data/flower_photos'

# define the transforms...
# resize, convert to tensors, ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# the initial entire dataset
dataset = datasets.ImageFolder(root_dir, transform=transform)

dataset_size = len(dataset)
print(f"Total number of images: {dataset_size}")

valid_size = int(valid_split*dataset_size)
train_size = len(dataset) - valid_size

# training and validation sets
train_data, valid_data = torch.utils.data.random_split(
    dataset, [train_size, valid_size]
)

print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(valid_data)}")

# training and validation data loaders
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_data, batch_size=batch_size, shuffle=False, num_workers=0
)