import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from prepareTheData import GalaxyDataset  # Import dataset class
from torchvision import models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure directories exist
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Setup TensorBoard with new log directory name
writer = SummaryWriter(log_dir="logs/galaxy_experiment")

# Define Transformations (with gentler augmentations)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Reduced rotation angle
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Use 224 for ResNet18
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=5),  # Reduced affine distortions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Filtered Dataset
df = pd.read_csv("filtered_dataset.csv")
full_dataset = GalaxyDataset(df, transform=transform)
print(f"Total images after filtering: {len(full_dataset)}")

# Split dataset (80% Train, 20% Validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoader Setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Define CNN Model (ResNet18 with Dropout)
class GalaxyCNN(nn.Module):
    def __init__(self, num_classes=3):  # Spiral, Elliptical, Irregular
        super(GalaxyCNN, self).__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        # Reduced dropout rate to 0.2 for stability
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Initialize Model
model = GalaxyCNN().to(device)

# Updated Class Weights based on new counts
class_counts = torch.tensor([4574, 2014, 7377], dtype=torch.float)
class_weights = torch.sqrt(1.0 / class_counts)  # Use square root to moderate the impact
class_weights /= class_weights.sum()  # Normalize
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Optimizer with lower learning rate and weight decay
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Learning Rate Scheduler: reduce LR on plateau based on validation loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# Training Loop
num_epochs = 40
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)

    # Validation Step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    # Log to TensorBoard
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save Checkpoint
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"{checkpoint_dir}/galaxy_checkpoint_{timestamp}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Step Learning Rate Scheduler based on validation loss
    scheduler.step(val_loss)

# Save Final Model
torch.save(model.state_dict(), "galaxy_final.pth")
print("Final model saved as 'galaxy_final.pth'")

# Close TensorBoard Writer
writer.close()
