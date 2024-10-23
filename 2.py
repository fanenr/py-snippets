import os
import pickle
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog


# 激活函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 辅助函数
get_inputs = (
    lambda values: (np.array(values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
)


# 神经网络
class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.lrate = learn_rate
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.wi_h = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.wh_o = np.random.rand(self.onodes, self.hnodes) - 0.5

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wi_h, inputs)
        hidden_outputs = sigmoid(hidden_inputs)
        outputs_inputs = np.dot(self.wh_o, hidden_outputs)
        outputs_outputs = sigmoid(outputs_inputs)
        output_errors = targets - outputs_outputs
        hidden_errors = np.dot(self.wh_o.T, output_errors)

        self.wh_o += self.lrate * np.dot(
            (output_errors * outputs_outputs * (1.0 - outputs_outputs)),
            np.transpose(hidden_outputs),
        )

        self.wi_h += self.lrate * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs),
        )

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wi_h, inputs)
        hidden_outputs = sigmoid(hidden_inputs)
        outputs_inputs = np.dot(self.wh_o, hidden_outputs)
        outputs_outputs = sigmoid(outputs_inputs)
        return outputs_outputs


# 参数定义
epochs = 5
input_nodes = 784
output_nodes = 10
hidden_nodes = 200
learning_rate = 0.1
trained_model = "trained_model.pkl"

# 加载模型
if os.path.exists(trained_model):
    with open(trained_model, "rb") as f:
        net = pickle.load(f)
    print(f"模型已从 {trained_model} 加载")
else:
    # 创建神经网络对象
    net = Network(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 加载训练数据集
    with open("mnist_train.csv", "r") as f:
        tran_list = f.readlines()

    # 开始训练
    for _ in range(epochs):
        for record in tran_list:
            values = record.split(",")
            inputs = get_inputs(values)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(values[0])] = 0.99
            net.train(inputs, targets)

    # 保存模型
    with open(trained_model, "wb") as f:
        pickle.dump(net, f)
    print(f"模型已保存到 {trained_model}")


# 加载测试数据集
with open("mnist_test.csv", "r") as f:
    test_list = f.readlines()

# 测试网络
scorecard = []

for record in test_list:
    values = record.split(",")
    inputs = get_inputs(values)
    outputs = net.query(inputs)

    if np.argmax(outputs) == int(values[0]):
        scorecard.append(1)
    else:
        scorecard.append(0)

# 计算正确率
array = np.asarray(scorecard)
print(f"正确率: {(array.sum() / array.size) * 100}%")


# 创建主窗口
window = tk.Tk()
window.title("MNIST")
window.geometry("600x400")

image = tk.Label(window)
image.pack()

result = tk.Label(window, text="")
result.pack()


# 识别图片
def identify_image():
    if not (file := filedialog.askopenfilename()):
        result.configure(text=f"请选择图片")
        return

    img = Image.open(file).convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_data = np.array(img).reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    outputs = net.query(img_data)
    label = np.argmax(outputs)

    # 显示图片
    img_tk = ImageTk.PhotoImage(img.resize((200, 200)))
    image.configure(image=img_tk)  # type: ignore
    image.image = img_tk  # type: ignore[attr-defined]

    # 显示预测结果
    result.configure(text=f"神经网络认为图中的数字是: {label}")


button = tk.Button(window, text="选择图片", command=identify_image)
button.pack()

window.mainloop()
