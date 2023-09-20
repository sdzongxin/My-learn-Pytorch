import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


EPOCH = 100
BATCH_SIZE = 64


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


dataset = DiabetesDataset('normalized_data.txt')
train_loader = DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(23, 12)
        self.linear2 = torch.nn.Linear(12, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# Note = open('2.txt', mode='w')

if __name__ == '__main__':
    model = Model()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    losses = []
    r2s = []
    mse = []
    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        running_r2 = 0.0
        running_mse = 0.0
        for i, data in enumerate(train_loader, 0):
            # 1. Prepare data
            inputs, labels = data  # GPU加速
            inputs = inputs
            labels = labels
            # 2. Forward
            y_pred = model(inputs)
            y_pred_data = y_pred.detach().numpy()     # 先回CPU
            labels_data = labels.detach().numpy()     # 先回CPU
            loss = criterion(y_pred, labels)
            running_loss += loss.item() / BATCH_SIZE
            running_mse += mean_squared_error(y_true=labels_data, y_pred=y_pred_data) / BATCH_SIZE
            running_r2 += r2_score(y_true=labels_data, y_pred=y_pred_data) / BATCH_SIZE
            print(epoch, i, y_pred_data - labels_data, loss.item(),
                  mean_squared_error(y_true=labels_data, y_pred=y_pred_data),
                  r2_score(y_true=labels_data, y_pred=y_pred_data))
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()
        epoch_loss = running_loss
        epoch_r2 = running_r2
        epoch_mse = running_mse
        losses.append(epoch_loss)
        r2s.append(epoch_r2)
        mse.append(epoch_mse)
    # Note.close()
    # 可视化损失曲线
    print(model.state_dict())
    fig1 = plt.plot(range(EPOCH), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    fig2 = plt.figure
    plt.plot(range(EPOCH), r2s)
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.show()
    fig3 = plt.figure
    plt.plot(range(EPOCH), mse)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.show()

    # # 输出模型的拟合结果
    # model.eval()
    # with torch.no_grad():
    #     predicted = model(inputs)
    # # 可视化拟合结果
    # plt.scatter(range(len(y)), y, label='Original data')
    # plt.scatter(range(len(predicted)), predicted, label='Fitted data', color='r')
    # plt.xlabel('Data points')
    # plt.ylabel('Output')
    # plt.legend()
    # plt.show()
