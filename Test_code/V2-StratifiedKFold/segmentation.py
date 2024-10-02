import os
import torch
from PIL import Image
import torchvision.transforms as transforms

# Hàm để dự đoán mask cho một ảnh dựa trên mô hình segmentation pretrained
def get_segmentation_mask(image_path, model):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply the necessary transforms (preprocessing)
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Send input to GPU if available
    input_tensor = input_tensor.to(device)

    # Predict mask
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = torch.argmax(output, dim=1).squeeze(0)

    return output_predictions.cpu().numpy()

# Hàm để lưu mask dưới dạng ảnh
def save_mask(mask, save_path):
    mask_image = Image.fromarray(mask.astype('uint8'), mode='L')
    mask_image.save(save_path)

# Hàm để xử lý tất cả ảnh trong một thư mục và các thư mục con
def process_images_in_subfolders(root_dir, output_root_dir, model):
    # Duyệt qua tất cả các thư mục con
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        if os.path.isdir(subfolder_path):  # Nếu đây là một thư mục
            output_subfolder = os.path.join(output_root_dir, subfolder)

            # Tạo thư mục đầu ra tương ứng nếu chưa có
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # Xử lý tất cả ảnh trong thư mục con
            for image_name in os.listdir(subfolder_path):
                if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Chỉ xử lý các file ảnh
                    image_path = os.path.join(subfolder_path, image_name)
                    print(f"Processing {image_path}...")

                    # Dự đoán mask
                    mask = get_segmentation_mask(image_path, model)

                    # Lưu mask dưới dạng ảnh
                    save_path = os.path.join(output_subfolder, f"mask_{image_name}")
                    save_mask(mask, save_path)
                    print(f"Saved mask to {save_path}")

# Load model pretrained (DeepLabV3 với backbone ResNet-101)
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()  # Đặt mô hình vào chế độ đánh giá (evaluation mode)

# Chuyển mô hình sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap the model with DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)

# Đường dẫn tới thư mục chứa các thư mục con với ảnh đầu vào
root_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_1/'  # Thay bằng đường dẫn tới thư mục chính
output_root_dir = '/kaggle/working/NLN-Loi-Phat/Dataset_3/'  # Thay bằng đường dẫn thư mục lưu mask

# Xử lý tất cả các ảnh trong các thư mục con và lưu mask
process_images_in_subfolders(root_dir, output_root_dir, model)
