from tqdm import tqdm
import pandas as pd
import torch

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