
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load extracted audio features
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "audio_features.npz"

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Normalize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Build Neural Network from scratch
class AudioNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


model = AudioNeuralNetwork().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
EPOCHS = 50

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()

with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)

y_true = y_test.cpu().numpy()
y_pred = predicted_classes.cpu().numpy()

print("\nAUDIO MODEL RESULTS")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save model and scaler
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.save(model.state_dict(), MODEL_DIR / "audio_model.pth")

import joblib
joblib.dump(scaler, MODEL_DIR / "audio_scaler.pkl")

print("\nAudio model and scaler saved successfully!")
