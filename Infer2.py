import torch
print(torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
print(torch.backends.mps.is_available(), "MPS" if torch.backends.mps.is_available() else "MPS not available")
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")  # 或者保持CPU

print("Using device:", device)

input_tensor = torch.randn(1, 10, 44100).to(device)

from gxl_ai_utils.utils import utils_file
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS available ✅")

    # 已分配的显存（字节数）
    allocated = torch.mps.current_allocated_memory()
    # 已保留的显存（字节数）
    reserved = torch.mps.driver_allocated_memory()

    print(f"Allocated: {allocated/1024/1024:.2f} MB")
    print(f"Reserved : {reserved/1024/1024:.2f} MB")
else:
    print("MPS not available ❌")

import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# ====== 设置设备 ======
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ====== 构造数据 ======
# 函数 y = sin(x)
N = 2000  # 样本数
x = torch.linspace(-10 * math.pi, 10 * math.pi, N).unsqueeze(1).to(device)
y = torch.sin(x)

# ====== 定义简单神经网络 ======
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 160),
    nn.ReLU(),
    nn.Linear(160, 1)
).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

utils_file.print_model_size(model)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS available ✅")

    # 已分配的显存（字节数）
    allocated = torch.mps.current_allocated_memory()
    # 已保留的显存（字节数）
    reserved = torch.mps.driver_allocated_memory()

    print(f"Allocated: {allocated/1024/1024:.2f} MB")
    print(f"Reserved : {reserved/1024/1024:.2f} MB")
else:
    print("MPS not available ❌")

# ====== 训练 ======
epochs = 20000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ====== 测试并可视化 ======
model.eval()
with torch.no_grad():
    y_pred = model(x)

x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()
y_pred_cpu = y_pred.cpu().numpy()

plt.figure(figsize=(8, 4))
plt.plot(x_cpu, y_cpu, label="True sin(x)")
plt.plot(x_cpu, y_pred_cpu, label="NN Prediction")
plt.legend()
plt.title("Neural Network Approximating sin(x)")
plt.show()
