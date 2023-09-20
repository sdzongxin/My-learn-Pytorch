import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import Dataset
from torchvision.transforms import ToTensor
from torchvision.models import resnet50


# 定义CPA模型
class CPA(nn.Module):
    def __init__(self, num_keypoints):
        super(CPA, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# 加载数据集
dataset = Dataset(root='path/to/dataset', transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型和优化器
model = CPA(num_keypoints=17)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数
criterion = nn.MSELoss()

# 训练模型
for epoch in range(10):
    for images, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print('Epoch:', epoch + 1, 'Loss:', loss.item())

# 在测试集上评估模型
test_dataset = Dataset(root='path/to/test/dataset', transform=ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=32)
total_loss = 0
with torch.no_grad():
    for images, targets in test_dataloader:
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
mse = total_loss / len(test_dataloader)
print("Mean Squared Error (MSE):", mse)
