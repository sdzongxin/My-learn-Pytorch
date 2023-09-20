import numpy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

EPOCH = 1000
BATCH_SIZE = 32
train_ratio = 0.8
test_ratio = 0.2
LR = 0.1


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(23, 36)
        self.linear2 = torch.nn.Linear(36, 100)
        self.linear3 = torch.nn.Linear(100, 16)
        self.linear4 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
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


dataset = DiabetesDataset('normalized_data.txt')
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True,
#                           num_workers=4)
# 计算划分的样本数量
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
# 使用random_split函数将数据集划分为训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 定义训练集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == '__main__':
    # 定义模型
    losses = []
    net_SGD = Model()
    net_Momentum = Model()
    net_RMSprop = Model()
    net_Adam = Model()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
    # criterion = torch.nn.MSELoss()
    # 定义优化器
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]  # 记录 training 时不同神经网络的 loss
    r2_score_his = [[], [], [], []]  # 记录 training 时不同net的决定系数r2
    mae_his = [[], [], [], []]  # 记录 training 时不同net的平均绝对误差
    rmse_his = [[], [], [], []]  # 记录 training 时不同net的均方根误差
    for epoch in range(EPOCH):
        # model.train()
        # 对每个优化器, 优化属于他的神经网络

        for i, (inputs, labels) in enumerate(train_loader):
            for net, opt, l_his, r_his, rootl_his in zip(nets, optimizers, losses_his, r2_score_his, rmse_his):
                output = net(inputs)  # get output for every net
                # inputs, labels output 都是float32 torch张量类型
                loss = loss_func(output, labels)  # compute loss for every net
                r2_score_solo = r2_score(labels, output)
                # mae = torch.nn.L1Loss(output, labels)
                # mse = mean_squared_error(labels.detach().numpy(), output.detach().numpy())
                rmse = torch.sqrt(loss)
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                opt.step()  # apply gradients
                l_his.append(loss.data.numpy())  # loss recoder
                # 将memoryview对象转换为NumPy数组
                r2_score_solo_array = np.asarray(r2_score_solo.detach().numpy())
                # 将NumPy数组转换为列表
                r_his.append(r2_score_solo_array.tolist())
                # r_his.append(r2_score_solo.numpy())
                rootl_his.append(rmse.data.numpy())
                print('Epoch: ', epoch, '| Step: ', i, loss.item(), r2_score_solo.item())

            # 1. Prepare data
            # inputs, labels = data  # GPU加速
            # criterion = torch.nn.MSELoss()
            # 2. Forward
            # y_pred = model(inputs)
            # y_pred_data = y_pred.detach().numpy()     # tensor -> numpy
            # labels_data = labels.detach().numpy()     # tensor -> numpy
            # loss = criterion(y_pred, labels)
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    fig1 = plt.figure(1)
    for j, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[j])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

    fig2 = plt.figure(2)
    for j, l_his in enumerate(r2_score_his):
        plt.plot(l_his, label=labels[j])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('R^2_score')
    plt.show()

    fig3 = plt.figure(3)
    for j, l_his in enumerate(rmse_his):
        plt.plot(l_his, label=labels[j])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('rmse_score')
    plt.show()

    # # 3. Backward
    # optimizer.zero_grad()
    # loss.backward()
    # # 4. Update
    # optimizer.step()
    # losses.append(loss.data.numpy())

    # 在测试集上计算MSE
    # total_loss = 0
    # total_r2s = 0
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         outputs = Model()
    #         loss = loss_func(outputs, targets)
    #         total_loss += loss.item()
    #         total_r2s += r2_score(y_true=targets.numpy(), y_pred=outputs.numpy())
    # plt.scatter(outputs.numpy(), targets.numpy(), label='comparing Data')
    # x = numpy.linspace(0.05, 0.6, 10)
    # y = x
    # plt.plot(x, y, 'r-',)
    # plt.xlabel('train_outputs')
    # plt.ylabel('real_outputs')
    # plt.title('Regression Model')
    # plt.legend()
    # plt.show()
    # mse = total_loss / len(test_loader)
    # r2s = total_r2s / len(test_loader)
    #
    # print("Mean Squared Error (MSE):", mse)
    # print("R2 score (R^2):", r2s)
    # w, b = Model.weight.data.numpy(), Model.bias.data.numpy()
    # print("训练好的权重：", w)
    # print("训练好的偏置：", b)
    # print(dir(Model))
