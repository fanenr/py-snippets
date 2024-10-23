import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

# 划分训练和测试集
data_train = datasets.MNIST(root="./3/", train=True, download=True, transform=transform)
data_test = datasets.MNIST(root="./3/", train=False, download=True, transform=transform)

data_train, _ = random_split(
    data_train, [1000, 59000], generator=torch.Generator().manual_seed(0)
)
data_test, _ = random_split(
    data_test, [1000, 9000], generator=torch.Generator().manual_seed(0)
)

# 创建数据加载器
train_loader = DataLoader(data_train, batch_size=4, shuffle=True)
test_loader = DataLoader(data_test, batch_size=4, shuffle=True)


# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 128, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x).view(-1, 7 * 7 * 128)
        return self.fc(x)


# 初始化
model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 5
for epoch in range(epochs):
    running_loss, running_correct, test_correct = 0.0, 0, 0

    # 训练
    for X_train, y_train in train_loader:
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += (outputs.argmax(1) == y_train).sum().item()

    # 测试
    with torch.no_grad():
        for X_test, y_test in test_loader:
            test_outputs = model(X_test)
            test_correct += (test_outputs.argmax(1) == y_test).sum().item()

    print(
        f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, "
        f"Train Acc: {100 * running_correct / len(data_train):.2f}%, "
        f"Test Acc: {100 * test_correct / len(data_test):.2f}%"
    )

# 预测结果
X_test, y_test = next(iter(test_loader))
pred = model(X_test).argmax(1)

print("Predicted:", pred.tolist())
print("Actual:", y_test.tolist())

img = torchvision.utils.make_grid(X_test).numpy().transpose(1, 2, 0)
img = img * 0.5 + 0.5
plt.imshow(img)
plt.show()
