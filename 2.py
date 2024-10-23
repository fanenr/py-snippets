import os
import pickle
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

# 激活函数
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# 辅助函数
get_inputs = (
    lambda values: (np.array(values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
)


# 神经网络
class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.lrate = learn_rate
        self.wi_h = np.random.rand(hidden_nodes, input_nodes) - 0.5
        self.wh_o = np.random.rand(output_nodes, hidden_nodes) - 0.5

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_outputs = sigmoid(np.dot(self.wi_h, inputs))
        outputs_outputs = sigmoid(np.dot(self.wh_o, hidden_outputs))

        output_errors = targets - outputs_outputs
        hidden_errors = np.dot(self.wh_o.T, output_errors)

        self.wh_o += self.lrate * np.dot(
            output_errors * outputs_outputs * (1 - outputs_outputs), hidden_outputs.T
        )
        self.wi_h += self.lrate * np.dot(
            hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T
        )

    def query(self, input_list):
        hidden_outputs = sigmoid(np.dot(self.wi_h, np.array(input_list, ndmin=2).T))
        return sigmoid(np.dot(self.wh_o, hidden_outputs))


# 参数定义
epochs, input_nodes, output_nodes, hidden_nodes, learning_rate = 5, 784, 10, 200, 0.1
trained_model = "trained_model.pkl"

# 加载或训练模型
if os.path.exists(trained_model):
    with open(trained_model, "rb") as f:
        net = pickle.load(f)
    print(f"模型已从 {trained_model} 加载")
else:
    net = Network(input_nodes, hidden_nodes, output_nodes, learning_rate)

    with open("mnist_train.csv", "r") as f:
        tran_list = f.readlines()

    for _ in range(epochs):
        for record in tran_list:
            values = record.split(",")
            inputs, targets = get_inputs(values), np.zeros(output_nodes) + 0.01
            targets[int(values[0])] = 0.99
            net.train(inputs, targets)

    with open(trained_model, "wb") as f:
        pickle.dump(net, f)
    print(f"模型已保存到 {trained_model}")

# 测试模型
scorecard = []

with open("mnist_test.csv", "r") as f:
    test_list = f.readlines()

for record in test_list:
    values = record.split(",")
    scorecard.append(
        1 if np.argmax(net.query(get_inputs(values))) == int(values[0]) else 0
    )

# 计算正确率
print(f"正确率: {np.mean(scorecard) * 100}%")


# 创建主窗口
def identify_image():
    if not (file := filedialog.askopenfilename()):
        result.configure(text=f"请选择图片")
        return

    img = Image.open(file).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    img_data = (np.array(img).reshape(784) / 255.0 * 0.99) + 0.01
    label = np.argmax(net.query(img_data))

    result.configure(text=f"神经网络认为图中的数字是: {label}")
    img_tk = ImageTk.PhotoImage(img.resize((200, 200)))
    image.configure(image=img_tk)  # type: ignore
    image.image = img_tk  # type: ignore[attr-defined]


window = tk.Tk()
window.title("MNIST")
window.geometry("600x400")

image = tk.Label(window)
image.pack()

result = tk.Label(window, text="")
result.pack()

tk.Button(window, text="选择图片", command=identify_image).pack()
window.mainloop()
