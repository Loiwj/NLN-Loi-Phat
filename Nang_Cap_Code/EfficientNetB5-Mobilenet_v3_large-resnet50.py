import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import copy
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR

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

# Load EfficientNet-B5, MobileNetV3 và ResNet50
efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
mobilenet = models.mobilenet_v3_large(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# Fine-tune cả ba mô hình
for param in efficientnet.parameters():
    param.requires_grad = True
for param in mobilenet.parameters():
    param.requires_grad = True
for param in resnet50.parameters():
    param.requires_grad = True

# Chỉnh sửa lớp đầu ra cuối cùng
num_ftrs_efficient = efficientnet._fc.in_features
efficientnet._fc = nn.Linear(num_ftrs_efficient, 512)

num_ftrs_mobilenet = mobilenet.classifier[-1].in_features
mobilenet.classifier[-1] = nn.Linear(num_ftrs_mobilenet, 512)

num_ftrs_resnet = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs_resnet, 512)

# Mô hình kết hợp
class CombinedModel(nn.Module):
    def __init__(self, efficientnet, mobilenet, resnet50, num_classes):
        super(CombinedModel, self).__init__()
        self.efficientnet = efficientnet
        self.mobilenet = mobilenet
        self.resnet50 = resnet50
        self.fc1 = nn.Linear(512 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        out1 = self.efficientnet(x)
        out2 = self.mobilenet(x)
        out3 = self.resnet50(x)
        combined_out = torch.cat((out1, out2, out3), dim=1)
        combined_out = torch.relu(self.bn1(self.fc1(combined_out)))
        combined_out = torch.relu(self.bn2(self.fc2(combined_out)))
        final_out = self.fc3(combined_out)
        return final_out

# Label Smoothing CrossEntropy
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

# Gradient Clipping
def clip_gradient(max_norm=2.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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
def train_model(model, criterion, optimizer, scheduler, early_stopping, num_epochs=200, use_cutmix=True, use_mixup=True, gradient_accumulation_steps=4):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    if use_cutmix:
                        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
                    elif use_mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)

                    inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        if phase == 'train' and (use_cutmix or use_mixup):
                            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                        else:
                            loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss = loss / gradient_accumulation_steps
                        scaler.scale(loss).backward()

                        if (i + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

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
            print(f'    Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(early_stopping.best_model_wts)
                    return model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Khởi tạo mô hình kết hợp
model = CombinedModel(efficientnet, mobilenet, resnet50, num_classes=len(dataset.classes)).to(device)

# Sử dụng DataParallel để sử dụng nhiều GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    
# Định nghĩa loss function và optimizer
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Cosine Annealing Scheduler với Warmup
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-12)

# Initialize EarlyStopping
early_stopping = EarlyStopping(patience=50, verbose=True)

scaler = GradScaler()

# Initialize SWA
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
swa_start = 150  # Example epoch to start SWA

# Huấn luyện mô hình
model = train_model(model, criterion, optimizer, scheduler, early_stopping, num_epochs=200)

# Update BN statistics for the SWA model at the end of training
torch.optim.swa_utils.update_bn(train_loader, swa_model)

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
    print(f'Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')

evaluate_model(model, dataloaders['val'])