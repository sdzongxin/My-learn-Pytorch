import torch
from sklearn.svm import SVR
"""
支持向量回归（Support Vector Regression，SVR）是一种使用支持向量机（Support Vector Machine，SVM）算法解决回归问题的方法。
在PyTorch中，可以使用SVR算法的变种——支持向量机回归（Support Vector Machine Regression，SVMR）来实现。
下面是一个使用SVMR进行回归的示例代码：
在上述示例中，首先创建了一个示例数据矩阵X和目标变量向量y，并将其转换为numpy数组。
然后，定义了一个SVR模型，使用sklearn库中的SVR类来实现。
在模型的参数中，设置了kernel参数为'rbf'表示使用径向基函数作为核函数，C参数为1.0表示正则化的强度，epsilon参数为0.1表示容忍度。
然后，使用fit方法进行模型训练。
最后，使用训练好的模型进行预测。
请注意，以上代码示例中的SVR模型只有一个输入特征和一个输出目标变量。如果你的数据具有多个特征，可以相应地调整输入和模型的维度。
"""
# 创建示例数据
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# 将数据转换为numpy数组
X = X.numpy()
y = y.numpy()

# 定义SVR模型
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 训练模型
model.fit(X, y)

# 使用训练好的模型进行预测
predicted = model.predict(X)
print("预测结果：", predicted)