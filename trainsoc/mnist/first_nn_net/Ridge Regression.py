import torch
import torch.nn as nn
import torch.optim as optim
"""
岭回归（Ridge Regression）是一种线性回归的方法，通过添加L2正则化项来解决多重共线性问题。
在PyTorch中，可以使用torch.nn库中的线性回归模型和优化器来实现岭回归。
下面是一个使用岭回归进行回归的示例代码：
在上述示例中，首先创建了一个示例数据矩阵X和目标变量向量y。
然后，定义了一个岭回归模型，使用nn.Linear类来实现。接下来，定义了损失函数和优化器，分别使用nn.MSELoss和optim.Ridge。
在优化器的参数中，设置了alpha参数来实现L2正则化，其中alpha的值越大，正则化的强度越大。
然后，使用循环进行模型训练，其中包括前向传播、计算损失、反向传播和优化步骤。最后，使用训练好的模型进行预测。
请注意，以上代码示例中的岭回归模型只有一个输入特征和一个输出目标变量。如果你的数据具有多个特征，可以相应地调整输入和模型的维度。
"""
# 创建示例数据
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# 定义岭回归模型
model = nn.Linear(1, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Ridge(model.parameters(), lr=0.01, alpha=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印每个epoch的损失
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 使用训练好的模型进行预测
predicted = model(X)
print("预测结果：", predicted.detach().numpy())