import torch
import random
from PIL import Image, ImageFilter

# Hyperparameters and device configuration
NUM_CLASSES = 20  
EPOCHS = 1000  
BATCH_SIZE = 32 
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class NumpyImageDataset:
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx].transpose((1, 2, 0))
        label = self.labels[idx]

        image = Image.fromarray((image * 255).astype('uint8'), 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, label