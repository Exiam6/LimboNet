from utils import device, NUM_CLASSES, BATCH_SIZE, learning_rate, momentum, weight_decay,EPOCHS
from tqdm import tqdm
import torch.optim as optim
from model import LabelSmoothingLoss
from dataset import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


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