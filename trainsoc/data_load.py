import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Excel_dataset(Dataset):

    def __init__(self, dir, if_normalize=False, if_onehot=False):
        super(Excel_dataset, self).__init__()

        if (dir.endswith('.csv')):
            data = pd.read_csv(dir)
        elif (dir.endswith('.xlsx') or dir.endswith('.xls')):
            data = pd.read_excel(dir, engine="openpyxl")

        nplist = data.T.to_numpy()
        data = nplist[0:-1].T
        self.data = np.float64(data)
        self.target = nplist[-1]

        self.target_type = []
        #记录标签有几类
        for i in self.target:
            try:
                self.target_type.index(i)
            except(BaseException):
                self.target_type.append(i)
                # print(i)
        # 将标签转为自然数编码
        self.target_num = []
        for i in self.target:
            self.target_num.append(self.target_type.index(i))
            # print(self.target_type.index(i))

        # Tensor化
        self.data = np.array(self.data)
        self.data = torch.FloatTensor(self.data)
        self.target_num = np.array(self.target_num)
        self.target_num = self.target_num.astype(float)
        self.target_num = torch.LongTensor(self.target_num)
        self.if_onehot = if_onehot
        #生成独热编码
        self.target_onehot = []
        if if_onehot == True:

            for i in self.target_num:
                tar = F.one_hot(i.to(torch.int64), len(self.target_type))
                self.target_onehot.append(tar)
            # pass

        if if_normalize == True:
            self.data = nn.functional.normalize(self.data)

    def __getitem__(self, index):
        # tar = F.one_hot(self.target[index].to(torch.int64), len(self.target_type))
        # print(tar)
        if self.if_onehot == True:
            return self.data[index], self.target_onehot[index]

        else:
            return self.data[index], self.target_num[index]

    def __len__(self):
        return len(self.target)


def data_split(data, rate):
    train_l = int(len(data) * rate)
    test_l = len(data) - train_l
    """打乱数据集并且划分"""
    train_set, test_set = torch.utils.data.random_split(data, [train_l, test_l])
    return train_set, test_set
