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
import copy

# Vô hiệu hóa cảnh báo FutureWarning,UserWarning, DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check if multiple GPUs are available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define transformations for the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.Resize((128, 128)),
        transforms.GaussianBlur(3),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
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

# Set up Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Early Stopping parameters
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased. Reset counter.')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# Combined model that takes the outputs of both EfficientNet and MobileNetV3
class CombinedModel(nn.Module):
    def __init__(self, efficientnet, mobilenet, num_classes):
        super(CombinedModel, self).__init__()
        self.efficientnet = efficientnet
        self.mobilenet = mobilenet
        self.fc1 = nn.Linear(512 * 2, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(512, num_classes)  # Final output layer

    def forward(self, x):
        out1 = self.efficientnet(x)
        out2 = self.mobilenet(x)
        combined_out = torch.cat((out1, out2), dim=1).contiguous()  # Ensure tensor is contiguous
        combined_out = self.fc1(combined_out)
        combined_out = self.bn1(combined_out)
        combined_out = torch.relu(combined_out)
        combined_out = self.dropout1(combined_out)
        combined_out = self.fc2(combined_out)
        combined_out = self.bn2(combined_out)
        combined_out = torch.relu(combined_out)
        combined_out = self.dropout2(combined_out)
        combined_out = self.fc3(combined_out)
        combined_out = self.bn3(combined_out)
        combined_out = torch.relu(combined_out)
        combined_out = self.dropout3(combined_out)
        combined_out = self.fc4(combined_out)
        combined_out = self.bn4(combined_out)
        combined_out = torch.relu(combined_out)
        combined_out = self.dropout4(combined_out)
        final_out = self.fc5(combined_out)
        return final_out

# Get training and validation indices
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f'Fold {fold + 1}')

    # Create training and validation subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Apply appropriate transformations to each subset
    train_subset.dataset.transform = data_transforms['train']
    val_subset.dataset.transform = data_transforms['val']

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

    # Load EfficientNet-B5 and EfficientNet-B6 models
    efficientnet_b5 = EfficientNet.from_pretrained('efficientnet-b5')
    efficientnet_b6 = EfficientNet.from_pretrained('efficientnet-b6')

    # Fine-tune EfficientNet-B5 and EfficientNet-B6
    for param in efficientnet_b5.parameters():
        param.requires_grad = True  # Unfreeze all layers in EfficientNet-B5

    for param in efficientnet_b6.parameters():
        param.requires_grad = True  # Unfreeze all layers in EfficientNet-B6

    # Modify the final layers of both models to match the number of classes
    num_ftrs_b5 = efficientnet_b5._fc.in_features
    efficientnet_b5._fc = nn.Linear(num_ftrs_b5, 512)

    num_ftrs_b6 = efficientnet_b6._fc.in_features
    efficientnet_b6._fc = nn.Linear(num_ftrs_b6, 512)

    # If more than 1 GPU is available, wrap the models in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        efficientnet_b5 = nn.DataParallel(efficientnet_b5)
        efficientnet_b6 = nn.DataParallel(efficientnet_b6)

    # Move the models to GPU(s)
    efficientnet_b5 = efficientnet_b5.to(device)
    efficientnet_b6 = efficientnet_b6.to(device)

    # Initialize the combined model
    model = CombinedModel(efficientnet_b5, efficientnet_b6, num_classes=len(dataset.classes)).to(device)

    # Sử dụng Optimizer AdamW
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Sử dụng scheduler ReduceLROnPlateau để giảm learning rate dựa trên validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    # Define the CutMix and MixUp techniques
    use_cutmix = True
    use_mixup = True

    # Training loop
    # Define the cutmix_data function
    def cutmix_data(inputs, labels, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(inputs.size()[0])
        target_a = labels
        target_b = labels[rand_index]
        
        # Select the bounding box region for CutMix
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        return inputs, target_a, target_b, lam

    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def mixup_data(inputs, labels, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        rand_index = torch.randperm(inputs.size()[0])
        mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
        target_a, target_b = labels, labels[rand_index]
        return mixed_inputs, target_a, target_b, lam


# Training loop
    def train_model(model, criterion, optimizer, scheduler, early_stopping, num_epochs=50, use_cutmix=True, use_mixup=True):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 30)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                all_preds = []
                all_labels = []

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Apply CutMix or Mixup during training phase
                    if phase == 'train':
                        if use_cutmix:
                            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
                        elif use_mixup:
                            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                        inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        # Calculate loss for CutMix or Mixup
                        if phase == 'train' and (use_cutmix or use_mixup):
                            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                        else:
                            loss = criterion(outputs, labels)

                        # Backward pass and optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Collect all predictions and labels for calculating metrics
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # Calculate loss and accuracy for this phase
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                f1 = f1_score(all_labels, all_preds, average='weighted')

                # Print the metrics for each phase
                print(f'  {phase.capitalize()} Phase:')
                print(f'    Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}')

                # Deep copy the model
                if phase == 'val':
                    scheduler.step(epoch_loss)
                    early_stopping(epoch_loss, model)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            print()

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Evaluate the model
    def evaluate_model(model, dataloader):
        model.eval()  # Đặt mô hình vào chế độ đánh giá (evaluation mode)
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():  # Tắt gradient để giảm bớt việc tính toán không cần thiết
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass: Dự đoán kết quả
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Thống kê số lượng dự đoán đúng
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        # Tính toán độ chính xác
        accuracy = running_corrects.double() / total_samples
        print(f'Accuracy of the model: {accuracy:.4f}')

        return accuracy

    # Define the loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, early_stopping, num_epochs=50)

    # Save the trained model for the current fold
    torch.save(model.state_dict(), f'combined_model_fold_{fold + 1}.pth')

    # Evaluate accuracy on the validation set
    evaluate_model(model, val_loader)
