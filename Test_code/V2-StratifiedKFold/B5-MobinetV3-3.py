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

# Vô hiệu hóa cảnh báo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), shear=10),
        transforms.GaussianBlur(3),
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

    # Load EfficientNet-B5 and MobileNetV3 models
    efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
    mobilenet = models.mobilenet_v3_large(pretrained=True)
    
    # Fine-tune thêm các lớp sâu hơn của EfficientNet và MobileNetV3
    for param in efficientnet.parameters():
        param.requires_grad = False  # Đóng băng các lớp đầu tiên

    # Fine-tune từ lớp 30 trở đi của EfficientNet
    for param in efficientnet._blocks[30:].parameters():
        param.requires_grad = True

    # Fine-tune lớp fully-connected cuối cùng
    for param in efficientnet._fc.parameters():
        param.requires_grad = True

    # Fine-tune từ các lớp middle trở đi của MobileNetV3
    for param in mobilenet.features[12:].parameters():
        param.requires_grad = True

    # Fine-tune lớp classifier cuối cùng của MobileNetV3
    for param in mobilenet.classifier.parameters():
        param.requires_grad = True


    # Modify the final layers of both models to match the number of classes
    num_ftrs_efficient = efficientnet._fc.in_features
    efficientnet._fc = nn.Linear(num_ftrs_efficient, 512)

    num_ftrs_mobilenet = mobilenet.classifier[-1].in_features
    mobilenet.classifier[-1] = nn.Linear(num_ftrs_mobilenet, 512)

    # If more than 1 GPU is available, wrap the models in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        efficientnet = nn.DataParallel(efficientnet)
        mobilenet = nn.DataParallel(mobilenet)

    # Move the models to GPU(s)
    efficientnet = efficientnet.to(device)
    mobilenet = mobilenet.to(device)

    # Combined model that takes the outputs of both EfficientNet and MobileNetV3
    class CombinedModel(nn.Module):
        def __init__(self, efficientnet, mobilenet, num_classes):
            super(CombinedModel, self).__init__()
            self.efficientnet = efficientnet
            self.mobilenet = mobilenet
            self.fc = nn.Linear(512 * 2, num_classes)  # Concatenating outputs from both models

        def forward(self, x):
            out1 = self.efficientnet(x)
            out2 = self.mobilenet(x)
            combined_out = torch.cat((out1, out2), dim=1)  # Concatenating along the feature dimension
            final_out = self.fc(combined_out)
            return final_out

    # Initialize the combined model
    model = CombinedModel(efficientnet, mobilenet, num_classes=len(dataset.classes)).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Sử dụng tốc độ học khác nhau cho các phần của mô hình
    optimizer = optim.AdamW([
        {'params': efficientnet.module._blocks[30:].parameters(), 'lr': 1e-5} if isinstance(efficientnet, nn.DataParallel) else {'params': efficientnet._blocks[30:].parameters(), 'lr': 1e-5},  # Lớp giữa học chậm hơn
        {'params': efficientnet.module._fc.parameters(), 'lr': 1e-3} if isinstance(efficientnet, nn.DataParallel) else {'params': efficientnet._fc.parameters(), 'lr': 1e-3},  # Lớp cuối học nhanh hơn
        {'params': mobilenet.module.features[12:].parameters(), 'lr': 1e-5} if isinstance(mobilenet, nn.DataParallel) else {'params': mobilenet.features[12:].parameters(), 'lr': 1e-5},  # Lớp giữa của MobileNet học chậm hơn
        {'params': mobilenet.module.classifier.parameters(), 'lr': 1e-3} if isinstance(mobilenet, nn.DataParallel) else {'params': mobilenet.classifier.parameters(), 'lr': 1e-3}  # Lớp cuối cùng của MobileNet học nhanh hơn
    ])

    
    # Điều chỉnh scheduler theo phương pháp giảm tốc độ học dựa trên mất mát
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    # Training loop
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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

                # Step the scheduler if in the validation phase
                if phase == 'val':
                    scheduler.step(epoch_loss)

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

    # Train the model
    model = train_model(model, criterion, optimizer,scheduler, num_epochs=50)

    # Save the trained model for the current fold
    torch.save(model.state_dict(), f'combined_model_fold_{fold + 1}.pth')

    # Evaluate accuracy on the validation set
    evaluate_model(model, val_loader)
