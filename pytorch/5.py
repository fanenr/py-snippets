import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(28, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])


# 实例化模型
model = RNN().to(device)

# 加载实验数据
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./5/", train=True, transform=transform, download=True),
    batch_size=64,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./5/", train=False, transform=transform, download=True),
    batch_size=64,
    shuffle=False,
)


# 准确率计算函数
def evaluate():
    model.eval()

    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            labels = labels.to(device)
            imgs = imgs.squeeze(1).to(device)
            predictions = model(imgs).argmax(1)
            correct += (predictions == labels).sum().item()
    print(f"Test ACC: {correct / len(test_loader):.5f}")

    model.train()


# 优化器和损失函数
epochs = 5
losser = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
print("开始训练...")
evaluate()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}\n{'-' * 10}")

    for idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.squeeze(1).to(device), labels.to(device)

        # 前向传播与损失计算
        loss = losser(model(imgs), labels)

        # 反向传播与更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f"Batch {idx}, Loss: {loss.item():.6f}")

    evaluate()

# 保存模型
torch.save(model, "rnn_model.pt")
