import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# class to index mapping (loaded from file)
base_dir = os.path.dirname(os.path.abspath(__file__))
classmap = json.loads(open(os.path.join(base_dir, "class_to_idx.txt")).read())


class MappedImageFolder(datasets.ImageFolder):
    """ImageFolder that uses the class_to_idx mapping from class_to_idx.txt."""

    def find_classes(self, directory):
        classes = sorted(classmap, key=classmap.get)
        return classes, classmap

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform
IMG_SIZE = 224
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # resize
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # normalize
    ]
)
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # resize
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # normalize
    ]
)

# data (train / val only; the test set is held out for evaluate.py)
data_dir = "/home/kan/Study/dl_architecture/competition2026/idrid_resplit_seed2026/idrid_resplit_seed2026"
train_dataset = MappedImageFolder(
    root=f"{data_dir}/train", transform=train_transform
)
val_dataset = MappedImageFolder(root=f"{data_dir}/val", transform=transform)
dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)


# model
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


num_classes = len(dataset.classes)
model = MyModel(num_classes).to(device)

# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# model training
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for idx, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(
        f"Epoch {epoch + 1}, Loss: {total_loss / total}, Accuracy: {100 * correct / total}%"
    )

    # validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch + 1}, Val Accuracy: {100 * val_correct / val_total}%")


model_scripted = torch.jit.script(model.to("cpu"))  # script
model_scripted.save("idrid_model.pth")  # save model
