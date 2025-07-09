# ====================================================
# train.py - 最终生产版
# ====================================================

# 1. 导入库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import re
import time
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm
# 2. CHLDataset 类 (我们已验证的最终版)
# train.py中新的、轻量化的CHLDataset类

class CHLDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        """
        初始化函数，现在只接收预处理好的数据文件夹路径。
        """
        super().__init__()
        # 获取所有预处理好的输入文件的路径，并排序
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pt')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.pt')])
        
        # 确保文件可以一一对应
        assert len(self.input_files) == len(self.label_files), "预处理后的输入和标签文件数量不匹配！"
        print(f"成功找到 {len(self.input_files)} 个预处理好的训练样本。")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        这个方法现在变得极其简单和快速！
        它只负责从硬盘加载两个已经处理好的小文件。
        """
        input_tensor = torch.load(self.input_files[idx])
        label_tensor = torch.load(self.label_files[idx])
        return input_tensor, label_tensor

# 3. 定义轻量级U-Net模型
class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个简单的卷积序列作为模型
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.layers(x)

# 4. 主执行部分
if __name__ == '__main__':
    # a. 设置参数
    CMEMS_FILE_PATH = r'/home/xly/xly/xly/cmemsData/September.nc' # 使用您确认过的绝对路径
    JAXA_DATA_DIR = 'jaxaData'
    LON_RANGE = [110, 120]
    LAT_RANGE = [18, 23]
    NUM_EPOCHS = 20 # 我们可以增加训练轮数
    BATCH_SIZE = 8  # 也可以适当增加批次大小
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # b. 准备数据
    print("\n正在准备数据...")
    dataset = CHLDataset(cmems_file_path=CMEMS_FILE_PATH, jaxa_dir=JAXA_DATA_DIR, lon_range=LON_RANGE, lat_range=LAT_RANGE)
    if len(dataset) > 0:
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print("数据准备完毕。")

        # c. 准备模型
        print("\n正在准备模型...")
        model = SimpleUnet().to(device)
        
        # 关键验证：检查模型是否有参数
        num_params = len(list(model.parameters()))
        if num_params > 0:
            print(f"模型创建成功，包含 {num_params} 组可训练参数。")
        else:
            raise ValueError("模型创建失败，参数列表为空！")

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("损失函数和优化器定义成功。")

        # d. 开始训练循环
        print("\n--- 开始训练 ---")
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            model.train()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_function(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"    - Epoch [{epoch+1}/{NUM_EPOCHS}], 平均损失 (Loss): {avg_epoch_loss:.6f}")

        end_time = time.time()
        print("\n--- 训练完成 ---")
        print(f"    - 总耗时: {((end_time - start_time) / 60):.2f} 分钟")
        print("模型正在保存权重")
        model_save_path = 'unet_model_weights.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已成功保存至: {model_save_path}")
    else:
        print("\n数据加载失败，训练未开始。")
