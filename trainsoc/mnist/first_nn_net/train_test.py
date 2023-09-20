import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

EPOCH = 100
BATCH_SIZE = 64
train_ratio = 0.8
test_ratio = 0.2
LR = 0.01


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(23, 64)
        self.bn = nn.BatchNorm1d(64)
        self.linear2 = torch.nn.Linear(64, 512)
        self.linear3 = torch.nn.Linear(512, 1)
        # self.linear4 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.bn(self.bn(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        # x = self.sigmoid(self.linear4(x))
        return x


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def r2_score(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2


dataset = DiabetesDataset('data.txt')

# 计算划分的样本数量
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
# 使用random_split函数将数据集划分为训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 定义训练集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 计算训练数据的均值和标准差
scaler = StandardScaler()
scaler.fit(train_dataset)
mean = scaler.mean_
std = scaler.scale_

Net = Model()

if __name__ == '__main__':
    # 定义模型
    net_SGD = Model()
    # 定义优化器
    opt = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    # 定义损失函数
    loss_func = torch.nn.MSELoss()
    for epoch in range(EPOCH):
        # model.train()
        # 对每个优化器, 优化属于他的神经网络
        for i, (inputs, labels) in enumerate(train_loader):
            # 进行批归一化
            normalized_inputs = (inputs - mean) / std

            output = Net(normalized_inputs)
            original_data = normalized_inputs * std + mean
            original_output = output * std + mean
            # inputs, labels output 都是float32 torch张量类型
            loss = loss_func(output, labels)  # compute loss for every net
            r2_score_solo = r2_score(labels, output)
            # mae = torch.nn.L1Loss(output, labels)
            # mse = mean_squared_error(labels.detach().numpy(), output.detach().numpy())
            rmse = torch.sqrt(loss)
            # l_his.append(loss.data.numpy())  # loss recoder
            # 将memoryview对象转换为NumPy数组
            r2_score_solo_array = np.asarray(r2_score_solo.detach().numpy())
            # 将NumPy数组转换为列表
            # r_his.append(r2_score_solo_array.tolist())
            # r_his.append(r2_score_solo.numpy())
            # rootl_his.append(rmse.data.numpy())
            print('Epoch: ', epoch, '| Step: ', i, loss.item(), r2_score_solo.item())
            opt.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            opt.step()  # apply gradients

    # 在测试集上计算MSE
    total_loss = 0
    total_r2s = 0
    with torch.no_grad():
        for inputs, targets in test_loader:

            normalized_test_data = (inputs - mean) / std

            Net = Model()
            outputs = Net(normalized_test_data)
            loss = loss_func(outputs, targets)
            total_loss += loss.item()
            total_r2s += r2_score(y_true=targets, y_pred=outputs)
            print(outputs.detach().numpy(), targets.detach().numpy())
    plt.scatter(outputs.numpy(), targets.numpy(), label='comparing Data')
    x = numpy.linspace(0, 1, 10)
    y = x
    plt.plot(x, y, 'r-',)
    plt.xlabel('train_outputs')
    plt.ylabel('real_outputs')
    plt.title('Regression Model')
    plt.legend()
    plt.show()
    mse = total_loss / len(test_loader)
    r2s = r2_score(y_true=targets, y_pred=outputs)
    print("Mean Squared Error (MSE):", mse)
    print("R2 score (R^2):", r2s)

