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
                 lon_range, lat_range, history_len=5, forecast_len=3, **kwargs):
        super().__init__()
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.sequence_len = history_len + forecast_len

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


#注意这里swinUnet的尺寸要求是固定的。除非去修改模型结构细节
#采取补白或者放缩操作，但补白会导致输出图像有效部分比例不协调
    def _pad_tensor(self, tensor):
        _, h, w = tensor.shape
        
        required_multiple = 224 

        pad_h = (required_multiple - h % required_multiple) % required_multiple
        pad_w = (required_multiple - w % required_multiple) % required_multiple

        padding = (0, pad_w, 0, pad_h)
        
        return F.pad(tensor, padding, "constant", 0)
        

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
# 3. 主执行部分
# ----------------------------------------------------
if __name__ == '__main__':
    # a. 配置服务器路径和核心参数
    #具体文件路径自己设置
    CMEMS_DATA_DIR = 
    JAXA_DATA_DIR = 
    UNET_BG_FIELD_DIR = 
    #第一个数据是CMEMS数据(完整无缺失数据集)
    #第二个数据是jaxa卫星数据
    #第三个是生成的背景场数据
    
    # 训练超参数
    NUM_EPOCHS = 200      
    BATCH_SIZE = 8        
    ACCUMULATION_STEPS = 8 
    LEARNING_RATE = 1e-6  
    
    # 数据参数
    HISTORY_LEN = 5
    FORECAST_LEN = 3
    LON_RANGE = [111, 116] 
    LAT_RANGE = [20, 23]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")
    
    # b. 参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch_window7_224_lite.yaml', help='path to config file')
    parser.add_argument('--opts', help="Modify config options.", default=None, nargs='+')
    args, _ = parser.parse_known_args()
    config = get_config(args)

    # c. 划分并准备数据
    print("\n正在准备Swin-Unet的训练数据...")
    all_cmems_files = sorted(glob.glob(os.path.join(CMEMS_DATA_DIR, '*.nc')))
    if len(all_cmems_files) < 2:
        raise ValueError("数据文件不足，至少需要2个文件才能划分训练集和验证集。")
    
    train_files = all_cmems_files[:-1]
    val_files = all_cmems_files[-1:]
    
    print(f"    - {len(train_files)} 个文件用于训练: {[os.path.basename(f) for f in train_files]}")
    print(f"    - {len(val_files)} 个文件用于验证: {[os.path.basename(f) for f in val_files]}")

    print("\n创建训练数据集...")
    train_dataset = SwinUnetForecastDataset(
        cmems_file_paths=train_files, jaxa_dir=JAXA_DATA_DIR, unet_bg_field_dir=UNET_BG_FIELD_DIR,
        lon_range=LON_RANGE, lat_range=LAT_RANGE, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN
    )
    print("\n创建验证数据集...")
    val_dataset = SwinUnetForecastDataset(
        cmems_file_paths=val_files, jaxa_dir=JAXA_DATA_DIR, unet_bg_field_dir=UNET_BG_FIELD_DIR,
        lon_range=LON_RANGE, lat_range=LAT_RANGE, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("数据准备完毕。")

    # d. 准备Swin-Unet模型
    print("\n正在准备Swin-Unet模型...")
    config.defrost()
    config.MODEL.SWIN.IN_CHANS = HISTORY_LEN * 2
    config.DATA.IMG_SIZE = (224, 224) 
    config.freeze()
    
    model = SwinUnet(config, num_classes=FORECAST_LEN).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    print("模型、损失函数、优化器和调度器定义成功。")

    # e. 开始训练循环 (包含验证和模型权重保存)
    print("\n--- 开始训练Swin-Unet---")
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training", leave=False)
        
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                predictions = model(inputs)
                loss = loss_function(predictions, labels)
                loss = loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * ACCUMULATION_STEPS
            train_progress_bar.set_postfix(train_loss=f"{loss.item() * ACCUMULATION_STEPS:.6f}")
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    predictions = model(inputs)
                    loss = loss_function(predictions, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        tqdm.write(f"    - Epoch [{epoch+1}/{NUM_EPOCHS}] | 训练Loss: {avg_train_loss:.6f} | 验证Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_save_path = f'best_model_epoch_{epoch+1}_valloss_{avg_val_loss:.4f}_{timestamp}.pth'
            torch.save(model.state_dict(), model_save_path)
            tqdm.write(f"    -> 验证Loss，模型已保存至: {model_save_path}")
            
        scheduler.step(avg_val_loss)

    end_time = time.time()
    print("\n--- 训练完成 ---")
    print(f"    - 总耗时: {((end_time - start_time) / 60):.2f} 分钟")



