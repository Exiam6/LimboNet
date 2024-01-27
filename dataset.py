import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import GaussianBlur, NumpyImageDataset

def load_data(batch_size, validation_split=0.01):

    transform = transforms.Compose([
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform2 = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5 
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1,2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform3 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = np.load('data/cifar_train_data.npy').astype(np.float32) / 255.0
    train_labels = np.load('data/cifar_train_label.npy')
    test_data = np.load('data/cifar_test_data.npy').astype(np.float32) / 255.0

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=validation_split, random_state=42)

    train_dataset = NumpyImageDataset(train_data, train_labels, transform=transform)
    train_dataset_DA = NumpyImageDataset(train_data, train_labels, transform=transform2)
    val_dataset = NumpyImageDataset(val_data, val_labels, transform=transform)
    test_dataset = NumpyImageDataset(test_data, np.zeros(len(test_data)), transform=transform)

    train_loader = DataLoader(train_dataset_DA, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader