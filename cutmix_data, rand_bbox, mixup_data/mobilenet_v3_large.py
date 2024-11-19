import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import warnings
import numpy as np
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchsummary import summary
from io import StringIO
import sys
import matplotlib.pyplot as plt

# Vô hiệu hóa cảnh báo FutureWarning, UserWarning, DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Kiểm tra GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Định nghĩa phép biến đổi cho tập dữ liệu
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), shear=10),
        transforms.RandomErasing(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Đường dẫn đến tập dữ liệu của bạn
image_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_1/'

# Tải tập dữ liệu
dataset = datasets.ImageFolder(root=image_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Load MobileNetV3 Large
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)

# Chỉnh sửa lớp đầu ra cuối cùng
num_ftrs_mobilenet = mobilenet_v3_large.classifier[3].in_features
mobilenet_v3_large.classifier[3] = nn.Linear(num_ftrs_mobilenet, 1024)

class CustomModel(nn.Module):
    def __init__(self, mobilenet_v3_large, num_classes):
        super(CustomModel, self).__init__()
        self.mobilenet_v3_large = mobilenet_v3_large
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        out1 = self.mobilenet_v3_large(x)
        combined_out = torch.relu(self.bn1(self.fc1(out1)))
        combined_out = self.dropout1(combined_out)
        combined_out = torch.relu(self.bn2(self.fc2(combined_out)))
        combined_out = self.dropout2(combined_out)
        combined_out = torch.relu(self.bn3(self.fc3(combined_out)))
        combined_out = self.dropout3(combined_out)
        final_out = self.fc4(combined_out)
        return final_out

# Instantiate the combined model
num_classes = len(dataset.classes)
model = CustomModel(mobilenet_v3_large, num_classes)

# Sử dụng DataParallel để sử dụng nhiều GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / targets.size(1)
        loss = (-targets * log_probs).mean(0).sum()
        return loss

# CutMix function
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x, target_a, target_b, lam

# Random bounding box
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# Mixup function
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
        
# Hàm huấn luyện
def train_model(model, criterion, early_stopping, optimizer, num_epochs=50, grad_clip=1.0):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Open a file to log the training process
    with open('mobilenet_v3_large_log.csv', 'w') as log_file:
        log_file.write('Epoch,Phase,Loss,Accuracy,Precision,Recall,F1-Score\n')
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
            
                    if phase == 'train':
                        # Randomly decide to apply CutMix or Mixup
                        rand = np.random.rand()
                        if rand < 0.5:
                            # Apply CutMix
                            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
                            with torch.amp.autocast(device_type=device.type):
                                outputs = model(inputs)
                                loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
                        else:
                            # Apply Mixup
                            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                            with torch.amp.autocast(device_type=device.type):
                                outputs = model(inputs)
                                loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
                        _, preds = torch.max(outputs, 1)
                        # Calculate correct predictions
                        preds_cpu = preds.cpu()
                        targets_a_cpu = targets_a.cpu()
                        targets_b_cpu = targets_b.cpu()
                        running_corrects += (lam * (preds_cpu == targets_a_cpu).sum().item() + (1 - lam) * (preds_cpu == targets_b_cpu).sum().item())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            
                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            running_corrects += torch.sum(preds == labels.data).item()
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
            
                # Calculate precision, recall, and F1-score for both phases
                if len(all_preds) > 0 and len(all_labels) > 0:
                    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                else:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
            
                log_file.write(f'{epoch+1},{phase},{epoch_loss:.4f},{epoch_acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n')
                print(f'  {phase.capitalize()} Phase:')
                # Print the metrics
                print(f'    Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}')

                if phase == 'val':
                    # Early stopping logic
                    early_stopping(epoch_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        model.load_state_dict(early_stopping.best_model_wts)
                        return model
                    # Track best accuracy model
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            print()

    print(f'Best val Acc: {best_acc:.4f}')
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Định nghĩa tiêu chuẩn và bộ tối ưu hóa
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()  # Initialize GradScaler for mixed precision
early_stopping = EarlyStopping(patience=25, verbose=True)
model = train_model(model, criterion, early_stopping, optimizer, num_epochs=200)

# Đánh giá mô hình
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    with open('mobilenet_v3_large_log.csv', 'a') as log_file:
        log_file.write('Evaluation Metrics:\n')
        log_file.write(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

evaluate_model(model, dataloaders['val'])


# Đánh giá mô hình trên tập dữ liệu kiểm tra
with open('mobilenet_v3_large_log.csv', 'a') as log_file:
    log_file.write('\n')
    log_file.write('Evaluation on Test Set:\n')
dataset_2_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_2/'
dataset_2 = datasets.ImageFolder(root=dataset_2_dir, transform=data_transforms['val'])
test_loader = DataLoader(dataset_2, batch_size=64, shuffle=False, num_workers=4)
print('Evaluation on Test Set:')
evaluate_model(model, test_loader)

# In ra ma trận nhầm lẫn (confusion matrix)


def plot_confusion_matrix(model, dataloader, classes, name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(name)

# Plot confusion matrix for validation set
plot_confusion_matrix(model, dataloaders['val'], dataset.classes, 'mobilenet_v3_val_cm.png')
# Plot confusion matrix for test set
plot_confusion_matrix(model, test_loader, dataset_2.classes, 'mobilenet_v3_test_cm.png')

# Capture the summary output
summary_str = StringIO()
sys.stdout = summary_str
summary(model, input_size=(3, 224, 224))
sys.stdout = sys.__stdout__
print(summary_str.getvalue())

# Extract the required lines
summary_lines = summary_str.getvalue().split('\n')
total_params = next(line for line in summary_lines if line.startswith('Total params')).replace(',', '')
trainable_params = next(line for line in summary_lines if line.startswith('Trainable params')).replace(',', '')

# Write the required lines to the file
with open('mobilenet_v3_large_log.csv', 'a') as f:
    f.write('Model Summary:\n')
    f.write(total_params + '\n')
    f.write(trainable_params + '\n')
