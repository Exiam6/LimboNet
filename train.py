import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import collections

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_CLASSES = 20  
EPOCHS = 1000  
BATCH_SIZE = 32 
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CobbBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        super(CobbBottleneck, self).__init__()
        D = out_channels // 4

        # 1x1 followed by 7x7 convolution branch
        self.conv1x1_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, D, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )
        # 1x1 followed by 3x3 convolution branch
        self.conv1x1_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # 1x1 followed by 5x5 convolution branch
        self.conv1x1_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, D, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # 3x3 maxpool followed by 1x1 convolution branch
        self.maxpool_conv1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1x1_7x7 = self.conv1x1_7x7(x) 
        out1x1_3x3 = self.conv1x1_3x3(x)
        out1x1_5x5 = self.conv1x1_5x5(x)
        out_maxpool_conv1x1 = self.maxpool_conv1x1(x)

        # Concatenate along the channel dimension
        out = torch.cat([out1x1_7x7, out1x1_3x3, out1x1_5x5, out_maxpool_conv1x1], dim=1)

        residual = self.shortcut(x)
        out += residual
        out = F.relu(out)

        return out

class LimboNet(nn.Module):
    def __init__(self, num_classes=20, cardinality=4, base_width=4, widen_factor=4):
        super(LimboNet, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.widen_factor = widen_factor

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
    
        self.stage_1 = self._make_stage(64, 128, 1)
        self.stage_2 = self._make_stage(128, 256, 2)  
        self.stage_3 = self._make_stage(256, 512, 2)  
        self.stage_4 = self._make_stage(512, 1024, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)  
        init.kaiming_normal_(self.classifier.weight)

    def _make_stage(self, in_channels, out_channels, stride):
        return CobbBottleneck(in_channels, out_channels, stride, self.cardinality, self.base_width, self.widen_factor)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

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

def train_and_validate(model, criterion, device, train_loader, val_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, total=len(val_loader), desc=f"Validating Epoch {epoch}"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.0f}%)')
    return val_loss, val_accuracy

def train_model(model, device, lr, momentum, T_max):
    train_loader, val_loader,_ = load_data(BATCH_SIZE,)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=0.1)
    #scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    scheduler = StepLR(optimizer,step_size=100,gamma=0.25)

    for epoch in range(1, EPOCHS + 1):
        train_and_validate(model, criterion, device, train_loader, val_loader, optimizer, epoch)
        scheduler.step()
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
            }
            torch.save(checkpoint, f"result/model_checkpoint_epoch_{epoch}.pth")

def predict(model, device, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
            images = data[0].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

def save_predictions_to_csv(predictions, file_name):
    df = pd.DataFrame({'id': range(len(predictions)), 'category': predictions})
    df.to_csv(file_name, index=False)
    
def load_checkpoint(model,filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, epoch, loss


model = LimboNet(num_classes=NUM_CLASSES , cardinality=4, base_width=4, widen_factor=4)
model.to(device)
train_model(model, device, learning_rate, momentum, T_max=EPOCHS)


torch.save(model, "Limbo.pth")
test_loader = load_data(BATCH_SIZE,)[2] 
predictions = predict(model, device, test_loader)
save_predictions_to_csv(predictions, 'predictions.csv')