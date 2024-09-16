import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Bước 3: Import các thư viện cần thiết
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os


# Đường dẫn tới thư mục chứa dữ liệu (ví dụ: '/content/drive/MyDrive/data')
data_dir = '/content/NLN-Loi-Phat/Dataset_2'

# Bước 5: Thiết lập các siêu tham số và cấu trúc dữ liệu
batch_size = 32
img_size = 224
num_epochs = 10
num_classes = 20
learning_rate = 0.001

# Chuẩn bị tập dữ liệu với augmentation và normalization
data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa như ResNet được huấn luyện
])

# Load tập dữ liệu và chia thành train/validation (80% train, 20% validation)
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Bước 6: Xây dựng mô hình ResNet18 với lớp đầu ra là 20 lớp (tùy thuộc vào số lớp của bạn)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Đưa mô hình lên GPU nếu có
model = model.to(device)

# Bước 7: Định nghĩa hàm mất mát và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Bước 8: Huấn luyện mô hình
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Vòng lặp huấn luyện
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs-1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Bước 9: Thực hiện huấn luyện
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Bước 10: Lưu mô hình sau khi huấn luyện
torch.save(model.state_dict(), '/content/drive/MyDrive/image_classification_model.pth')
