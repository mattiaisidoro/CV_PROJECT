# train_resnet.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import sys
import os

# Aggiunge la directory genitore (CV_PROJECT) al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_logic.preprocessing import PreprocessTransform

# === Dummy Dataset === (sostituisci con il tuo)
class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)  # Preprocessing + ToTensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# === Load ResNet50 and modify for grayscale ===
def get_model(num_classes=2):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Adatta il primo layer per immagini a 1 canale (invece di 3)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Congela tutti i layer
    for param in model.parameters():
        param.requires_grad = False

    # Sblocca solo il classificatore finale
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# === Esempio di utilizzo ===
if __name__ == "__main__":
    # Simula dati
    image_paths = ['shared_logic\\NORMAL(0).jpg', 'shared_logic\\PNEUMONIA(1).jpeg']
    labels = [0, 1]  # 0 = normale, 1 = polmonite

    # Preprocessing (incluso nel dataset)
    transform = PreprocessTransform(show=False)

    dataset = ChestXrayDataset(image_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

    # Esempio di un batch
    for epoch in range(5):
        model.train()
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
