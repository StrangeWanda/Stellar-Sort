import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Ensure CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure checkpoint directory exists
checkpoint_dir = "Inbw"
os.makedirs(checkpoint_dir, exist_ok=True)

class Galaxy10CNN(nn.Module):
        def __init__(self, num_classes=10):
            super(Galaxy10CNN, self).__init__()
            self.model = models.resnet18(weights="IMAGENET1K_V1")  # Updated parameter
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            return self.model(x)


# Define Galaxy10 Dataset
class Galaxy10Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        # Load dataset
        with h5py.File(self.h5_path, "r") as f:
            self.images = np.array(f["images"])  # Shape: (21785, 128, 128, 3)
            self.labels = np.array(f["ans"])     # Shape: (21785,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (128, 128, 3)
        label = self.labels[idx]

        # Convert to PIL Image and apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Define Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Run main training loop safely
if __name__ == "__main__":
    # Load Dataset
    data_path = os.path.expanduser("~/.astroNN/datasets/Galaxy10_DECals.h5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    full_dataset = Galaxy10Dataset(data_path, transform=transform)

    # Split dataset into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders (Use num_workers=0 for Windows)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize Model
    model = Galaxy10CNN().to(device)

    # Loss and Optimizer --- was changed retroactivly
    class_counts = torch.tensor([877, 1480, 2116, 1619, 263, 1607, 1464, 2112, 1140, 1510], dtype=torch.float)
    class_weights = 1.0 / class_counts  # Inverse weighting
    class_weights /= class_weights.sum()  # Normalize
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TensorBoard Setup
    writer = SummaryWriter(log_dir="logs/galaxy10")

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
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        # Validation
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
        checkpoint_path = f"{checkpoint_dir}/galaxy10_checkpoint_{timestamp}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Save Final Model
    torch.save(model.state_dict(), "galaxy10_final.pth")
    print("Final model saved as 'galaxy10_final.pth'")

    writer.close()
