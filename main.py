from model import LimboNet
from train import train_model
from predict import predict, save_predictions_to_csv
from utils import device, NUM_CLASSES, BATCH_SIZE
from dataset import load_data
import torch

model = LimboNet(num_classes=NUM_CLASSES , cardinality=4, base_width=4, widen_factor=4)
model.to(device)
train_model(model, device)

torch.save(model, "Limbo.pth")
test_loader = load_data(BATCH_SIZE,)[2] 
predictions = predict(model, device, test_loader)
save_predictions_to_csv(predictions, 'predictions.csv')
