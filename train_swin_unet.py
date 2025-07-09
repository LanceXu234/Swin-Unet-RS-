#'/home/xly/xly/xly/cmemsData/September.nc'
#'/home/xly/xly/xly/jaxaData'
#'/home/xly/xly/xly/background'

# ====================================================
# train_swin_unet.py - 最终生产版 (包含所有修正)
# ====================================================

# 1. 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import re
import time
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm
import argparse

# 导入Swin-Unet模型所需的文件
from vision_transformer import SwinUnet
from config import get_config 

print("--- Swin-Unet训练脚本启动 (最终版) ---")

# ----------------------------------------------------
# 2. 定义SwinUnetForecastDataset类 (最终版 - 带自动补白功能)
# ----------------------------------------------------
class SwinUnetForecastDataset(Dataset):
    def __init__(self, cmems_file_path, jaxa_dir, unet_bg_field_dir, 
                 lon_range, lat_range, history_len=5, forecast_len=3, window_size=7):
        super().__init__()
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.sequence_len = history_len + forecast_len
        self.window_size = window_size

        print("--- SwinUnet Dataset 初始化 ---")
        self.ds_cmems_full = xr.open_dataset(cmems_file_path, engine='h5netcdf')
        cmems_dates = {pd.to_datetime(t).strftime('%Y%m%d') for t in self.ds_cmems_full['time'].values}
        
        def _parse_date(f):
            m = re.search(r'(\d{8})', f)
            return m.group(0) if m else None
            
        self.jaxa_map = { _parse_date(f): os.path.join(jaxa_dir, f) for f in os.listdir(jaxa_dir) if _parse_date(f) and ".02401_02401" in f }
        self.bg_field_map = { _parse_date(f): os.path.join(unet_bg_field_dir, f) for f in os.listdir(unet_bg_field_dir) if _parse_date(f) }

        print("正在寻找所有可用的连续日期序列...")
        common_dates = sorted(list(cmems_dates & set(self.jaxa_map.keys()) & set(self.bg_field_map.keys())))
        
        self.valid_start_indices = []
        if len(common_dates) >= self.sequence_len:
            common_dates_dt = [pd.to_datetime(d) for d in common_dates]
            for i in range(len(common_dates_dt) - self.sequence_len + 1):
                is_continuous = all(common_dates_dt[i+j+1] == common_dates_dt[i+j] + pd.Timedelta(days=1) for j in range(self.sequence_len - 1))
                if is_continuous:
                    self.valid_start_indices.append(i)
        
        self.common_dates_lookup = common_dates

        if not self.valid_start_indices:
             raise ValueError("[错误] 未找到任何长度足够的连续数据序列用于训练。")
        else:
            print(f"成功找到 {len(self.valid_start_indices)} 个可用的训练序列。")

    def __len__(self):
        return len(self.valid_start_indices)

    
    def _pad_tensor(self, tensor):
        _, h, w = tensor.shape

        # For a Swin Transformer with patch_size=4, window_size=7, and 4 layers (like yours),
        # the image dimensions must be a multiple of 224.
        required_multiple = 224 

        pad_h = (required_multiple - h % required_multiple) % required_multiple
        pad_w = (required_multiple - w % required_multiple) % required_multiple

        # (pad_left, pad_right, pad_top, pad_bottom)
        padding = (0, pad_w, 0, pad_h)
        # Use "constant" padding to fill with zeros
        return F.pad(tensor, padding, "constant", 0)

       # _, h, w = tensor.shape
        #pad_h = (self.window_size - h % self.window_size) % self.window_size
        #pad_w = (self.window_size - w % self.window_size) % self.window_size
        #pad_h = 0
        #pad_w = 0
        #padding = (0, pad_w, 0, pad_h)
        #return F.pad(tensor, padding, "constant", 0)

    def _process_one_day_input(self, date_str):
        with xr.open_dataset(self.jaxa_map[date_str], engine='h5netcdf') as ds_jaxa:
            jaxa_lat_slice = slice(self.lat_range[1], self.lat_range[0])
            jaxa_lon_slice = slice(*self.lon_range)
            ds_jaxa_cropped = ds_jaxa.sel(longitude=jaxa_lon_slice, latitude=jaxa_lat_slice)
            jaxa_chl_cleaned = ds_jaxa_cropped['chlor_a'].where(ds_jaxa_cropped['chlor_a'] > 0)
            input_jaxa_np = np.nan_to_num(jaxa_chl_cleaned.values, nan=0.0)
            input_jaxa_tensor = torch.from_numpy(input_jaxa_np).unsqueeze(0).float()
        
        input_bg_field_tensor = torch.load(self.bg_field_map[date_str])
        day_tensor_2_channels = torch.cat([input_jaxa_tensor, input_bg_field_tensor.unsqueeze(0)], dim=0)
        return day_tensor_2_channels

    def _load_cmems_label(self, date_str):
        target_date_dt = pd.to_datetime(date_str)
        ds_cmems_daily = self.ds_cmems_full.sel(time=target_date_dt, method='nearest').isel(depth=0)
        ds_cmems_cropped = ds_cmems_daily.sel(longitude=slice(*self.lon_range), latitude=slice(*self.lat_range))
        label_np = np.nan_to_num(ds_cmems_cropped['chl'].values, nan=0.0)
        return torch.from_numpy(label_np).unsqueeze(0).float()

    def __getitem__(self, idx):
        start_index = self.valid_start_indices[idx]
        
        history_tensors = [self._process_one_day_input(self.common_dates_lookup[start_index + i]) for i in range(self.history_len)]
        input_X_unpadded = torch.cat(history_tensors, dim=0)

        future_tensors = [self._load_cmems_label(self.common_dates_lookup[start_index + self.history_len + i]) for i in range(self.forecast_len)]
        label_Y_unpadded = torch.cat(future_tensors, dim=0)

        input_X = self._pad_tensor(input_X_unpadded)
        label_Y = self._pad_tensor(label_Y_unpadded)
        
        return input_X, label_Y

# ----------------------------------------------------
# 3. 主执行部分 (最终版)
# ----------------------------------------------------
if __name__ == '__main__':
    # a. 创建一个功能完备的参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--accumulation-steps', type=int, default=0)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O0')
    parser.add_argument('--tag', default='test')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--batch-size', type=int, help='batch size for training')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    # b. 设置我们自己的路径和超参数
    CMEMS_FILE_PATH = '/home/xly/xly/xly/cmemsData/September.nc'
    JAXA_DATA_DIR = '/home/xly/xly/xly/jaxaData'
    UNET_BG_FIELD_DIR = '/home/xly/xly/xly/background'
    
    NUM_EPOCHS = 50
    BATCH_SIZE = args.batch_size if args.batch_size else config.DATA.BATCH_SIZE
    LEARNING_RATE = 1e-4
    HISTORY_LEN = 5
    FORECAST_LEN = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # c. 准备数据
    print("\n正在准备Swin-Unet的训练数据...")
    dataset = SwinUnetForecastDataset(
        cmems_file_path=CMEMS_FILE_PATH, jaxa_dir=JAXA_DATA_DIR,
        unet_bg_field_dir=UNET_BG_FIELD_DIR, lon_range=[110, 120],
        lat_range=[18, 23], history_len=HISTORY_LEN, forecast_len=FORECAST_LEN,
        window_size=config.MODEL.SWIN.WINDOW_SIZE
    )
    if len(dataset) > 0:
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print("数据准备完毕。")

        # d. 准备Swin-Unet模型
        print("\n正在准备Swin-Unet模型...")
        # 提前计算补白后的尺寸
        sample_input, _ = dataset[0]
        # 输入张量的形状是 (C, H, W)，所以高度是索引1，宽度是索引2
        h_padded, w_padded = sample_input.shape[1], sample_input.shape[2]
        print(f"补白后的数据尺寸: ({h_padded}, {w_padded})，将用于模型初始化。")
        
        config.defrost()
        config.MODEL.SWIN.IN_CHANS = HISTORY_LEN * 2
        config.DATA.IMG_SIZE = (h_padded, w_padded)
        config.freeze()
        
        num_classes = FORECAST_LEN
        model = SwinUnet(config, num_classes=num_classes).to(device)
        print("Swin-Unet模型创建成功。")

        loss_function = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print("损失函数和优化器定义成功。")

        # e. 开始训练循环
        print("\n--- 开始训练Swin-Unet ---")
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_function(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
            avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            tqdm.write(f"    - Epoch [{epoch+1}/{NUM_EPOCHS}] 结束, 平均损失 (Loss): {avg_epoch_loss:.6f}")
        
        end_time = time.time()
        print("\n--- 训练完成 ---")
        print(f"    - 总耗时: {((end_time - start_time) / 60):.2f} 分钟")

        print("\n正在保存Swin-Unet模型权重...")
        model_save_path = 'swin_unet_model_weights.pth' 
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已成功保存至: {model_save_path}")
    else:
        print("\n数据加载失败，训练未开始。")