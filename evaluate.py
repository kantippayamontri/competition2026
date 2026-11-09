import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch import nn
import sys,os
from PIL import Image
import json

from torch.utils.data.dataloader import default_collate

class SimpleDataset(Dataset):
    def __init__(self, directory, classmap, transform=None):
        self.directory = directory
        self.transform = transform
        # search subfolders too: directory/<class>/<image>.png
        self.images = [os.path.join(root, file)
                       for root, _, files in os.walk(directory)
                       for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.classmap = classmap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        l = os.path.basename(os.path.dirname(image_path))  # label = parent folder name

        if l in self.classmap:
            label = self.classmap[l]
            return image, label  # normal return
        else:
            print(f"Label for {l} not found in classmap. Skipping this item.")
            return None  # skip this item


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform (must match the eval transform in train.py)
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # resize
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # normalize
])

# data loader
base_dir = os.path.dirname(os.path.abspath(__file__))
classmap = json.loads(open(os.path.join(base_dir, 'class_to_idx.txt')).read())

class MappedImageFolder(datasets.ImageFolder):
    """ImageFolder that uses the class_to_idx mapping from class_to_idx.txt."""
    def find_classes(self, directory):
        classes = sorted(classmap, key=classmap.get)
        return classes, classmap

data_dir = os.path.join(base_dir, 'aid002-a3463-7ef7d056/aid002-a3463-7ef7d056/IDRiD_data_seed42_fold0/val')
dataset = MappedImageFolder(root=data_dir, transform=transform)
# print(dataset.class_to_idx) # class to index mapping

def custom_collate_fn(batch):
    # filter out Nones
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()  # return empty tensors
    return default_collate(batch)

test_dataset = dataset  # training mode: evaluate on the val split
test_loader = DataLoader(test_dataset, batch_size=16)
# print(classmap) # class to index mapping

# model_path = 'idrid_model.pth'
if len(sys.argv) > 2 and sys.argv[2] != 'training':
     # hidden mode: evaluate on the hidden test set
     testds = SimpleDataset('/home/kan/Study/dl_architecture/competition2026/test_idrid', classmap,transform=transform)
     test_loader = DataLoader(testds, batch_size=16, collate_fn=custom_collate_fn)
model_path = sys.argv[1]
model = torch.jit.load(model_path, map_location=device)

# Function to evaluate the model and identify misclassifications
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    misclassified = []  # List to store misclassified examples

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Handle multiple predictions and labels in a batch
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    label_name = dataset.classes[labels[i].item()]
                    predicted_name = dataset.classes[predicted[i].item()]
                    misclassified.append((label_name, predicted_name))

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}))

    # Print misclassified examples
    # print("Misclassified examples:")
    # for label, pred in misclassified:
    #     print(f"Actual: {label}, Predicted: {pred}")

# Evaluate the model
evaluate_model(model, test_loader)
