
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_DIR / "data" / "metadata.csv"
FRAME_DIR = PROJECT_DIR / "data" / "processed" / "frames"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(METADATA_PATH)

# Split by video, so frames from one video
# cannot appear in both training and testing.
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_id"]
)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class FrameDataset(Dataset):
    def __init__(self, dataframe):
        self.samples = []

        for _, row in dataframe.iterrows():
            video_name = Path(row["video_path"]).stem
            folder = FRAME_DIR / row["label"] / video_name

            for image_path in folder.glob("*.jpg"):
                self.samples.append(
                    (image_path, int(row["label_id"]))
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        return image, label


train_dataset = FrameDataset(train_df)
test_dataset = FrameDataset(test_df)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False
)

print("Training frames:", len(train_dataset))
print("Testing frames:", len(test_dataset))


# CNN built from scratch
class VideoCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = VideoCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.001
)

# Training
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )


# Evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nVIDEO CNN RESULTS")
print("Frame-level accuracy:",
      accuracy_score(all_labels, all_predictions))

print("\nClassification Report:")
print(classification_report(
    all_labels, all_predictions, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

# Save model
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    MODEL_DIR / "video_model.pth"
)

print("\nVideo CNN model saved successfully!")
