import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt')
features = data[:, :23]
labels = data[:, -1].reshape(-1, 1)  # 将标签转换为列向量

# 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 进行主成分分析
pca = PCA()
features_pca = pca.fit_transform(features_scaled)

# 划分训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(features_pca, labels, test_size=0.2,
                                                                            random_state=42)

# 转换为张量
train_inputs = torch.tensor(train_features, dtype=torch.float32)
train_targets = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_features, dtype=torch.float32)
test_targets = torch.tensor(test_labels, dtype=torch.float32)


# 定义回归模型
class RegressionModel(nn.Module):
    def __init__(self, num_features):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


# 初始化回归模型
model = RegressionModel(num_features=features_pca.shape[1])

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        loss = criterion(outputs, test_targets)
        test_losses.append(loss.item())

    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, num_epochs, train_losses[-1],
                                                                        test_losses[-1]))

# 绘制训练集和测试集的损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 计算测试集的评价回归效果参数
model.eval()
with torch.no_grad():
    outputs = model(test_inputs)
    mse = mean_squared_error(test_targets, outputs.numpy())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets, outputs.numpy())
    r2 = r2_score(test_targets, outputs.numpy())
    print('MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}'.format(mse, rmse, mae, r2))