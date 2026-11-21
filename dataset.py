import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=224):
    """
    Returns training and validation transforms.
    Input size is 224x224 matching standard VGG16/MobileNet input specifications.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform

class TransformedDataset(Dataset):
    """Wraps a Subset or Dataset with a specific transform."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_cat_dog_data(data_dir="archive/animals", img_size=224, batch_size=32, train_split=0.8, seed=42):
    """
    Loads dataset from archive/animals (with subfolders 'cat' and 'dog'),
    splits into train and validation sets, and returns DataLoaders.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # Base dataset without transforms to allow separate train/val augmentations
    base_dataset = datasets.ImageFolder(root=data_dir)
    classes = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx

    total_len = len(base_dataset)
    train_len = int(train_split * total_len)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_len, val_len], generator=generator)

    train_transform, val_transform = get_transforms(img_size=img_size)

    # Wrap subsets with respective transforms
    # ImageFolder raw item returns (PIL Image, label)
    class CustomSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            path, target = self.subset.dataset.samples[self.subset.indices[idx]]
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, target

        def __len__(self):
            return len(self.subset)

    train_ds = CustomSubset(train_subset, train_transform)
    val_ds = CustomSubset(val_subset, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, classes, class_to_idx
