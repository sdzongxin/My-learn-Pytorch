import torch
import torch.nn as nn
import torch.optim as optim
"""
集成回归（Ensemble Regression）是一种通过组合多个回归模型来提高预测性能的方法。
在PyTorch中，可以使用torch.nn库中的模型和优化器来实现集成回归。下面是一个使用集成回归进行回归的示例代码：
在上述示例中，首先创建了一个示例数据矩阵X和目标变量向量y。
然后，定义了多个回归模型，使用nn.Linear类来实现。接下来，定义了损失函数和优化器，分别使用nn.MSELoss和optim.SGD。
然后，使用循环进行模型训练，其中包括前向传播、计算损失、反向传播和优化步骤。最后，使用训练好的模型进行预测，将多个模型的预测结果求平均作为最终的预测结果。
请注意，以上代码示例中的回归模型只有一个输入特征和一个输出目标变量。如果你的数据具有多个特征，可以相应地调整输入和模型的维度。同时，你也可以根据需要增加或减少回归模型的数量。
"""
# 创建示例数据
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# 定义多个回归模型
model1 = nn.Linear(1, 1)
model2 = nn.Linear(1, 1)
model3 = nn.Linear(1, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
optimizer3 = optim.SGD(model3.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播和计算损失
    outputs1 = model1(X)
    loss1 = criterion(outputs1, y)
    outputs2 = model2(X)
    loss2 = criterion(outputs2, y)
    outputs3 = model3(X)
    loss3 = criterion(outputs3, y)

    # 反向传播和优化
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    optimizer3.zero_grad()
    loss3.backward()
    optimizer3.step()

    # 打印每个epoch的损失
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'.format(epoch + 1, num_epochs, loss1.item(),
                                                                                  loss2.item(), loss3.item()))

# 使用训练好的模型进行预测
predicted1 = model1(X)
predicted2 = model2(X)
predicted3 = model3(X)
predicted = (predicted1 + predicted2 + predicted3) / 3
print("预测结果：", predicted.detach().numpy())