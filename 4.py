import os
import time
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

# 数据预处理与加载
data_dir = "./4/"
phases = ["train", "valid"]

data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transform)
    for x in phases
}
data_loaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
    for x in phases
}

# 加载预训练模型 ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 冻结参数并修改全连接层
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# 损失函数和优化器
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)

# 训练和验证模型
epochs = 1
start = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}\n{'-' * 10}")

    for phase in phases:
        model.train(phase == "train")
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in data_loaders[phase]:
            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects / len(image_datasets[phase])
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print(f"程序运行时间: {int((time.time() - start) / 60)}分钟")

# 结果可视化
X_example, Y_example = next(iter(data_loaders["train"]))
y_pred = model(X_example)
_, y_pred_class = torch.max(y_pred, 1)

example_classes = image_datasets["train"].classes
print("实际标签:", [example_classes[i] for i in Y_example])
print("预测标签:", [example_classes[i] for i in y_pred_class])

# 显示图片
img = torchvision.utils.make_grid(X_example).cpu().numpy().transpose([1, 2, 0])
plt.imshow(img)
plt.show()
