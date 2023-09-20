import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('data.txt')
features = data[:, :23]
labels = data[:, -1].reshape(-1, 1)  # 将标签转换为列向量

# 划分训练集和测试集
train_features = features[:4000]
train_labels = labels[:4000]
test_features = features[4000:]
test_labels = labels[4000:]

# 计算训练集的均值和标准差
scaler = StandardScaler()
scaler.fit(train_features)

# 将训练集和测试集转换为张量
train_inputs = torch.tensor(scaler.transform(train_features), dtype=torch.float32)
train_targets = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(scaler.transform(test_features), dtype=torch.float32)
test_targets = torch.tensor(test_labels, dtype=torch.float32)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 定义回归模型
class RegressionModel(nn.Module):
    def __init__(self, num_features):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.sigmoid(self.fc5(x))
        return x


num_epochs = 200
LR = 0.003
# 初始化回归模型
model = RegressionModel(num_features=23)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)
# optimizer= torch.optim.SGD(model.parameters(), lr=LR, momentum=0.8)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

# 训练模型

train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = 0.0
    test_loss = 0.0
    model.train()
    for inputs, targets in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # 计算训练集的平均损失
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 计算测试集的损失
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        test_loss = criterion(outputs, test_targets)
        test_losses.append(test_loss.item())

    # 打印损失值
    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss, test_loss))

plt.figure(1)
plt.scatter(outputs, test_targets, marker='o',edgecolor='lightblue', facecolor='none')
x = np.linspace(min(test_targets), max(test_targets), 10)
y = x
plt.plot(x, y, 'r-')
plt.xlabel('y_real_data')
plt.ylabel('y_prediction')
plt.show()
# 绘制训练集和测试集的损失曲线
plt.figure(2)
plt.plot(train_losses, label='Train Loss')
# plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 计算测试集的评价回归效果参数
model.eval()
with torch.no_grad():
    outputs = model(test_inputs)
    mse = criterion(outputs, test_targets)
    rmse = torch.sqrt(mse)
    mae = nn.L1Loss()(outputs, test_targets)
    r2 = 1 - mse / torch.var(test_targets)
    print('MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}'.format(mse.item(), rmse.item(), mae.item(), r2.item()))