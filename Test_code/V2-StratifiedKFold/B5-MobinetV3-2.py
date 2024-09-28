import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Disable FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Transformations for the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(),
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

# Path to your dataset
image_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_1/'  # Update this path

# Load dataset
dataset = datasets.ImageFolder(root=image_dir, transform=data_transforms['train'])

# Get class indices and targets
targets = dataset.targets

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training and validation indices
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f'Fold {fold + 1}')

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Apply transformations
    train_subset.dataset.transform = data_transforms['train']
    val_subset.dataset.transform = data_transforms['val']

    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

    # Load models
    efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
    mobilenet = models.mobilenet_v3_large(pretrained=True)

    # Modify EfficientNet
    num_ftrs_efficient = efficientnet._fc.in_features
    efficientnet._fc = nn.Sequential(
        nn.Linear(num_ftrs_efficient, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 20)  # 20 classes
    )

    # Modify MobileNet
    num_ftrs_mobilenet = mobilenet.classifier[-1].in_features
    mobilenet.classifier = nn.Sequential(
        nn.Linear(num_ftrs_mobilenet, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 20)  # 20 classes
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        efficientnet = nn.DataParallel(efficientnet)
        mobilenet = nn.DataParallel(mobilenet)

    # Move models to GPU
    efficientnet = efficientnet.to(device)
    mobilenet = mobilenet.to(device)

    # Combined model
    class CombinedModel(nn.Module):
        def __init__(self, efficientnet, mobilenet):
            super(CombinedModel, self).__init__()
            self.mobilenet = mobilenet
            self.efficientnet = efficientnet
            self.fc = nn.Linear(40, 20)  # 40 from concatenated outputs

        def forward(self, x):
            out1 = self.efficientnet(x)
            out2 = self.mobilenet(x)
            combined_out = torch.cat((out1, out2), dim=1)
            final_out = self.fc(combined_out)
            return final_out

    model = CombinedModel(efficientnet, mobilenet).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 30)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                f1 = f1_score(all_labels, all_preds, average='weighted')

                print(f'  {phase.capitalize()} Phase:')
                print(f'    Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
                print(f'    Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}')

                if phase == 'train':
                    scheduler.step()

    # Evaluation function
    def evaluate_model(model, dataloader):
        model.eval()
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        accuracy = running_corrects.double() / total_samples
        print(f'Accuracy of the model: {accuracy:.4f}')

        return accuracy

    # Train the model
    train_model(model, criterion, optimizer, scheduler, num_epochs=40)

    # Save the model
    torch.save(model.state_dict(), f'combined_model_fold_{fold + 1}.pth')

    # Evaluate the model
    evaluate_model(model, val_loader)