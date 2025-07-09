import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import re
from tqdm import tqdm

# --- 1. 定义我们之前训练时使用的SimpleUnet模型结构 ---
#    (必须和训练时的结构一模一样）
class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.layers(x)

# --- 2. 创建一个只读取JAXA原始数据的简单Dataset ---
class JAXADataset(Dataset):
    def __init__(self, jaxa_dir, lon_range, lat_range):
        super().__init__()
        self.lon_range = lon_range
        self.lat_range = lat_range
        
        # 使用我们之前验证过的文件名筛选逻辑
        self.jaxa_files = sorted([
            os.path.join(jaxa_dir, f) for f in os.listdir(jaxa_dir)
            if ".02401_02401" in f and f.endswith('.nc')
        ])
        print(f"找到 {len(self.jaxa_files)} 个JAXA文件准备进行处理。")

    def __len__(self):
        return len(self.jaxa_files)

    def __getitem__(self, idx):
        jaxa_filepath = self.jaxa_files[idx]
        with xr.open_dataset(jaxa_filepath, engine='h5netcdf') as ds_jaxa_full:
            # 裁剪并清洗数据
            jaxa_lat_slice = slice(self.lat_range[1], self.lat_range[0])
            jaxa_lon_slice = slice(*self.lon_range)
            ds_jaxa_cropped = ds_jaxa_full.sel(longitude=jaxa_lon_slice, latitude=jaxa_lat_slice)
            
            jaxa_chl_cleaned = ds_jaxa_cropped['chlor_a'].where(ds_jaxa_cropped['chlor_a'] > 0)
            
            # 只返回带真实空缺的输入
            input_np = np.nan_to_num(jaxa_chl_cleaned.values, nan=0.0)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).float()
            
            # 同时返回文件名，方便我们保存结果
            original_filename = os.path.basename(jaxa_filepath)
            
            return input_tensor, original_filename

# --- 3. 主执行部分 ---
if __name__ == '__main__':
    # a. 设置参数
    JAXA_DATA_DIR = r'C:\Users\12953\Desktop\iGEM\jaxaData'
    WEIGHTS_PATH = 'unet_model_weights.pth' # 我们刚刚训练保存的权重文件
    OUTPUT_DIR = './data/unet_background_fields/' # 创建一个新文件夹存放结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    LON_RANGE = [110, 120]
    LAT_RANGE = [18, 23]
    BATCH_SIZE = 8 # 可以根据您的GPU显存调整
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # b. 加载模型
    print(f"\n正在从 {WEIGHTS_PATH} 加载已训练的U-Net模型...")
    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval() # 切换到评估模式，这非常重要！
    print("模型加载成功。")

    # c. 准备数据
    print("\n正在准备JAXA原始数据...")
    dataset = JAXADataset(jaxa_dir=JAXA_DATA_DIR, lon_range=LON_RANGE, lat_range=LAT_RANGE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # 推理时不需要打乱顺序
    print("数据准备完毕。")

    # d. 执行推理并保存
    print("\n--- 开始生成背景场 ---")
    with torch.no_grad(): # 推理时不需要计算梯度
        for inputs, filenames in tqdm(data_loader, desc="Generating Background Fields"):
            inputs = inputs.to(device)
            
            # 模型进行预测（填补空缺）
            predictions = model(inputs)
            
            # 将结果从GPU移回CPU，并保存为新的.nc文件
            for i in range(predictions.size(0)):
                # 获取单个预测结果
                single_prediction = predictions[i].squeeze().cpu().numpy()
                # 获取对应的原始文件名
                original_filename = filenames[i]
                
                # 创建一个新的xarray.DataArray来保存结果，以便保留坐标信息（简化版）
                # 注意：一个更完整的实现会从原始文件中复制坐标信息
                save_path = os.path.join(OUTPUT_DIR, f"bg_field_{original_filename}")
                
                # 为了简单起见，我们直接保存为PyTorch Tensor
                torch.save(torch.from_numpy(single_prediction), save_path.replace('.nc', '.pt'))

    print(f"\n[成功] 所有背景场已生成并保存至: {OUTPUT_DIR}")