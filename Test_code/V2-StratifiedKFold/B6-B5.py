import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import copy

# Vô hiệu hóa cảnh báo FutureWarning, UserWarning, DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Kiểm tra GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Sử dụng thiết bị: {device}')

# Định nghĩa phép biến đổi cho tập dữ liệu
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), shear=10),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Đường dẫn đến tập dữ liệu của bạn
image_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_1/'  # Cập nhật đường dẫn

# Tải tập dữ liệu
dataset = datasets.ImageFolder(root=image_dir, transform=data_transforms['train'])

# Chia 70% cho tập train, 30% cho tập val
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Áp dụng phép biến đổi tương ứng cho từng tập
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

# Tạo DataLoader cho tập huấn luyện và tập kiểm tra
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Load EfficientNet-B6 model
efficientnet = EfficientNet.from_pretrained('efficientnet-b6')

# Fine-tune EfficientNet
for param in efficientnet.parameters():
    param.requires_grad = True  # Mở khóa toàn bộ EfficientNet

# Modify the final layers of EfficientNet-B6
num_ftrs_efficient = efficientnet._fc.in_features
efficientnet._fc = nn.Linear(num_ftrs_efficient, 512)

# Sử dụng nhiều GPU nếu có
if torch.cuda.device_count() > 1:
    print(f"Sử dụng {torch.cuda.device_count()} GPU!")
    efficientnet = nn.DataParallel(efficientnet)

# Chuyển mô hình sang GPU
efficientnet = efficientnet.to(device)

# Model class cho việc sử dụng chỉ EfficientNet-B6
class EfficientNetB6Model(nn.Module):
    def __init__(self, efficientnet, num_classes):
        super(EfficientNetB6Model, self).__init__()
        self.efficientnet = efficientnet
        self.fc1 = nn.Linear(512 , 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.attention1 = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.attention2 = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        combined_out = self.fc1(x)
        combined_out = self.bn1(combined_out)
        combined_out = torch.relu(combined_out)
        combined_out = combined_out.unsqueeze(0)
        combined_out, _ = self.attention1(combined_out, combined_out, combined_out)
        combined_out = combined_out.squeeze(0)
        combined_out = self.dropout1(combined_out)
        combined_out = self.fc2(combined_out)
        combined_out = self.bn2(combined_out)
        combined_out = torch.relu(combined_out)
        combined_out = combined_out.unsqueeze(0)
        combined_out, _ = self.attention2(combined_out, combined_out, combined_out)
        combined_out = combined_out.squeeze(0)
        combined_out = self.dropout2(combined_out)
        final_out = self.fc3(combined_out)
        return final_out

# Early Stopping parameters
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
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

# Định nghĩa hàm Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Khởi tạo mô hình
model = EfficientNetB6Model(efficientnet, num_classes=len(dataset.classes)).to(device)

# Định nghĩa loss function và optimizer
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
early_stopping = EarlyStopping(patience=10, verbose=True)

# Hàm huấn luyện mô hình (giữ lại phần CutMix và MixUp nếu cần)
def train_model(model, criterion, optimizer, scheduler, early_stopping, num_epochs=50):
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

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f'Best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, criterion, optimizer, scheduler, early_stopping, num_epochs=50)
