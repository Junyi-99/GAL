import torch
import torch.nn as nn
import numpy as np
from datasets import Letter
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(16, 100)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(100, 26)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

model = MyModel()

batch_size = 512
letter = Letter(root='', split='train', typ='corr', val='1.0')
dataloader = DataLoader(letter, batch_size=batch_size)


# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 定义 tensorboard writer 对象
writer = SummaryWriter()

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    running_loss = 0.
    correct_predictions = 0
    total_predictions = 0
    for i, data in enumerate(dataloader):
        # 获取数据和标签
        X_batch, y_batch = data['data'], data['target']
        
        # 正向传播
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        running_loss += loss.item()
        
        # 反向传播并更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(y_pred.data, 1)
        total_predictions += y_batch.size(0)
        correct_predictions += (predicted == y_batch).sum().item()
       
    # 打印损失和准确率
    accuracy = 100 * correct_predictions / total_predictions
    print("Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch, running_loss/len(dataloader), accuracy))

    # 将进度写入 tensorboard
    writer.add_scalar('Training/Loss', running_loss/len(dataloader), epoch)
    writer.add_scalar('Training/Accuracy', accuracy, epoch)


# 关闭 tensorboard writer 对象
writer.close()

# 预测新数据
X_new = torch.randn(10, 16)
y_pred = model(X_new)
_, predicted = torch.max(y_pred.data, 1)

# 输出预测结果
print("Predicted classes: ", predicted)