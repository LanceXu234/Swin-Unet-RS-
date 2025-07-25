# ====================================================
# test_and_visualize.py - 模型测试与可视化脚本
# ====================================================

# 1. 导入库
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import argparse
import os

# 导入您自己的项目文件(同一目录下文件可以共享)
from vision_transformer import SwinUnet
from config import get_config
from train_swin_unet import SwinUnetForecastDataset 

print("--- Swin-Unet 测试与可视化脚本启动 ---")

def visualize_comparison(prediction_tensor, ground_truth_tensor, day_index, lon_range, lat_range, save_path):
    """
    可视化模型预测与真实值的对比图。
    
    参数:
    - prediction_tensor: 单个预测结果的张量 (H, W)
    - ground_truth_tensor: 单个真实标签的张量 (H, W)
    - day_index: 预测的是未来第几天 (从0开始)
    - lon_range: 经度范围 [min_lon, max_lon]
    - lat_range: 纬度范围 [min_lat, max_lat]
    - save_path: 图片保存路径
    """

    # 反向变换 exp(x) - 1
    prediction_tensor_restored = torch.expm1(prediction_tensor)
    ground_truth_tensor_restored = torch.expm1(ground_truth_tensor)


    #张量转化为Numpy数组
    pred_np = prediction_tensor_restored.cpu().numpy()
    true_np = ground_truth_tensor_restored.cpu().numpy()

    #计算误差
    error_np=pred_np-true_np
    # 创建经纬度网格
    # 注意：这里的 '224' 是您补白后的大小，您需要确保这里的尺寸与数据尺寸一致
    lons = np.linspace(lon_range[0], lon_range[1], pred_np.shape[1])
    lats = np.linspace(lat_range[0], lat_range[1], pred_np.shape[0])
    
    # 共享的色阶范围
    vmax = np.percentile([true_np, pred_np], 99.5)
    vmin = np.min([true_np, pred_np]) # 最小值通常为0

    if vmax - vmin < 1e-5:
        vmax=vmin+1e-5
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # 2. 为 "误差图" 计算一个以0为中心的对称色阶
    max_abs_error = np.max(np.abs(error_np))
    if max_abs_error < 1e-5:
        max_abs_error = 1e-5
    error_norm = colors.TwoSlopeNorm(vmin=-max_abs_error, vcenter=0, vmax=max_abs_error)


    #创建1行3列的画布
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    fig.suptitle(f'Forecast Day {day_index + 1} - Chlorophyll Concentration', fontsize=16)

    #绘制真实值
    ax1=axes[0]
    mesh1 = ax1.pcolormesh(lons, lats, true_np, transform=ccrs.PlateCarree(), cmap='viridis', norm=norm)
    ax1.set_title('Ground Truth')

    #绘制预测值
    ax2=axes[1]
    mesh2 = ax2.pcolormesh(lons, lats, pred_np, transform=ccrs.PlateCarree(), cmap='viridis', norm=norm)
    ax2.set_title('Model Prediction')

    #绘制误差图
    ax3 = axes[2]
    # 误差图使用发散色图，中心为0
    #error_norm = colors.TwoSlopeNorm(vmin=error_np.min(), vcenter=0, vmax=error_np.max())
    mesh3 = ax3.pcolormesh(lons, lats, error_np, transform=ccrs.PlateCarree(), cmap='coolwarm', norm=error_norm)
    ax3.set_title('Error (Prediction - Truth)')

    # 为每个子图添加地理特征
    for ax in axes:
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

    # 添加颜色条
    fig.colorbar(mesh1, ax=axes[:2], orientation='horizontal', fraction=0.05, pad=0.1, label='Chlorophyll (mg/m^3)')
    fig.colorbar(mesh3, ax=ax3, orientation='horizontal', fraction=0.05, pad=0.1, label='Error')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path)
    print(f"可视化结果图已保存至: {save_path}")
    plt.close()       

if __name__ == '__main__':
    #配置参数和路径根据实际情况修改
    MODEL_WEIGHTS_PATH = 'swin_unet_weights_20250718-172212.pth'
    CONFIG_FILE_PATH = 'configs/swin_tiny_patch4_window7_224_lite.yaml'  
    # 数据路径 (与训练时相同)
    CMEMS_FILE_PATH = 
    JAXA_DATA_DIR = 
    UNET_BG_FIELD_DIR = 

    #模型参数(和训练的时候严格保持一致)  
    HISTORY_LEN=5
    FORECAST_LEN=3
    LON_RANGE=[111,116]
    LAT_RANGE=[20,23]

    #输出设置
    OUTPUT_DIR='./test_results'
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备：{device}")

    # ----------------------------------------------------
    # b. 加载模型
    # ----------------------------------------------------
    print("\n正在准备Swin-Unet模型...")
    # 解析Swin-Unet的配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=CONFIG_FILE_PATH)
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--accumulation-steps', type=int, default=0)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O0')
    parser.add_argument('--tag', default='test')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')

    # 关键：添加缺失的 batch-size 参数 
    parser.add_argument('--batch-size', type=int, help='batch size for training/testing')
    args, _ = parser.parse_known_args()
    config = get_config(args)
    #=================================================
    # 手动覆盖模型输入通道数，确保与训练时创建的模型结构完全一致
    # 训练时我们使用了 5天 * 2个通道/天 = 10个输入通道
    config.defrost()
    config.MODEL.SWIN.IN_CHANS = HISTORY_LEN * 2
    config.freeze()
    #==================================================

    #创建模型结构
    model = SwinUnet(config, num_classes=FORECAST_LEN).to(device)

    # 加载已训练好的权重
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    print(f"成功加载模型权重: {MODEL_WEIGHTS_PATH}")

    #模型变更为评估模式，关闭dropout等
    model.eval()

    # ----------------------------------------------------
    # c. 准备测试数据
    # ----------------------------------------------------   
    print("\n正在准备测试数据...")     
    test_dataset = SwinUnetForecastDataset(
        cmems_file_paths=[CMEMS_FILE_PATH], jaxa_dir=JAXA_DATA_DIR,
        unet_bg_field_dir=UNET_BG_FIELD_DIR, lon_range=LON_RANGE,
        lat_range=LAT_RANGE, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN,
        window_size=config.MODEL.SWIN.WINDOW_SIZE
    )
    
    # 使用 DataLoader 加载数据，batch_size=1 便于逐个处理和可视化
    # shuffle=False 保证每次测试的结果都一样
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if len(test_dataset) == 0:
        raise ValueError("未找到可用的测试数据序列。")
    print(f"数据准备完毕，共找到 {len(test_dataset)} 个可测试样本。")

    # ----------------------------------------------------
    # d. 进行推理、评估和可视化
    # ----------------------------------------------------
    print("\n--- 开始进行推理和可视化 ---")
    total_mse = 0    

    #禁用梯度计算
    with torch.no_grad():
        for i,(inputs,labels)in enumerate(tqdm(test_loader, desc="Testing Samples")):
            inputs, labels = inputs.to(device), labels.to(device)

            #模型预测
            predictions=model(inputs)

            #计算并累计MSE损失
            mse = F.mse_loss(predictions, labels)
            total_mse += mse.item()

            #-----------可视化比对-------------
            for day in range(FORECAST_LEN):
                pred_single_day=predictions[0,day,:,:]#取第0个batch第day个预测
                label_single_day=labels[0,day,:,:]#取第0个batch第day个标签
                
                save_filename=os.path.join(OUTPUT_DIR, f'sample_{i}_forecast_day_{day+1}.png')

                visualize_comparison(
                    prediction_tensor=pred_single_day,
                    ground_truth_tensor=label_single_day,
                    day_index=day,
                    lon_range=LON_RANGE,
                    lat_range=LAT_RANGE,
                    save_path=save_filename

                )

            print("\n演示完成，仅测试第一个样本。如需测试全部，请注释掉脚本中的 'break'。")
            
    
    avg_mse = total_mse / (i + 1)
    print(f"\n--- 测试完成 ---")
    print(f"在测试的 {i + 1} 个样本上，平均均方误差 (MSE) 为: {avg_mse:.6f}")

    







