
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = BASE_DIR / "data" / "processed" / "audio_features.npz"
VIDEO_PATH = BASE_DIR / "data" / "processed" / "video_features.npz"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load features
audio_data = np.load(AUDIO_PATH)
video_data = np.load(VIDEO_PATH)

X_audio = audio_data["X"]
X_video = video_data["X"]
y = audio_data["y"]

# Check labels and combine features
assert np.array_equal(y, video_data["y"]), "Audio/video labels do not match!"

X = np.concatenate([X_audio, X_video], axis=1)

print("Combined feature shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / "multimodal_scaler.pkl")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Multimodal neural network from scratch
class MultimodalModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel(X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.001
)

# Train
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Evaluate
model.eval()

with torch.no_grad():
    predictions = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

print("\nMultimodal Model Results")
print("Accuracy:", accuracy_score(y_test.numpy(), predictions))
print("\nClassification Report:")
print(classification_report(
    y_test.numpy(),
    predictions,
    target_names=["Truthful", "Deceptive"],
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test.numpy(), predictions))

# Save model
torch.save(model.state_dict(), MODEL_DIR / "multimodal_model.pth")

print("\nMultimodal model saved successfully!")
