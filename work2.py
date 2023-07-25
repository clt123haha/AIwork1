import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
'''最终要获得一个十通道的输出'''
# 超参数设置
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Mnist数据集
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 定义简单的神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  #全连接层   [batch_size, in_features] -》  [batch_size, out_features]
        self.relu = nn.ReLU() #激励函数  保持shape不变
        self.fc2 = nn.Linear(hidden_size, num_classes)  #全连接层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 定义模型、损失函数和优化器
model = NeuralNet(784, 100, 10)
criterion = nn.CrossEntropyLoss()  #损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #优化器

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):  #训练五轮
    for i, (images, labels) in enumerate(train_loader):
        # 将图像数据展平
        images = images.reshape(-1, 28 * 28)  #-1自动计算

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  #梯度设置为0
        loss.backward()
        optimizer.step()

        #打印进度
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 测试模型
model.eval()  #测试模式
with torch.no_grad():  #关闭梯度
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  #得到概率最大的
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('准确率: {} %'.format(100 * correct / total))
