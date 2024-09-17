import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import random
import shutil

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Function to split dataset into train/val
def split_dataset(image_dir, output_dir, train_ratio=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for folder in [train_dir, val_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    for class_folder in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        train_class_dir = os.path.join(train_dir, class_folder)
        val_class_dir = os.path.join(val_dir, class_folder)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        images = os.listdir(class_path)
        random.shuffle(images)

        split_point = int(len(images) * train_ratio)
        train_images = images[:split_point]
        val_images = images[split_point:]

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

    print(f"Dataset split into {train_dir} and {val_dir}")

# Path to your image dataset (before split)
image_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_2/'  # Update this path

# Path where the split dataset will be stored
output_dir = '/kaggle/working/NLN-Loi-Phat/data'  # Update this path

# Split dataset (80% train, 20% val)
split_dataset(image_dir, output_dir, train_ratio=0.8)

# Define transformations for the training and validation data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Path to your data folders
data_dir = '/kaggle/working/NLN-Loi-Phat/data'

# Load the data
image_datasets = {
    'train': datasets.ImageFolder(root=f'{data_dir}/train', transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=f'{data_dir}/val', transform=data_transforms['val'])
}

# Create DataLoader
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}

# Get dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Initialize EfficientNet-B0 (pre-trained on ImageNet)
model = models.efficientnet_b0(pretrained=True)

# Modify the final layer to match the number of classes in your dataset
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_ftrs, len(image_datasets['train'].classes))
)

# Move the model to GPU if available
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Calculate precision, recall, and F1-score
            precision = running_corrects.double() / (running_corrects.double() + (dataset_sizes[phase] - running_corrects.double()))
            recall = running_corrects.double() / dataset_sizes[phase]
            f1 = 2 * (precision * recall) / (precision + recall)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1:.4f}')


        print()

    return model

# Train the model
model = train_model(model, criterion, optimizer, num_epochs=25)

# Save the trained model
torch.save(model.state_dict(), 'efficientnet_b0_finetuned.pth')
