"""
Seismic preprocessing helper.

End-to-end workflow:
1. Run this script to slice patches from matdata.mat:
   python prepare_data.py
2. Train the diffusion model on the generated patches:
   python scripts/image_train.py --data_path qiepian/marmousi_patches.npy
3. Sample reconstructions with the seismic config:
   python main.py --config_file configs/seismic.yaml
"""

import argparse
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

def slice_seismic_data(
    mat_path="matdata.mat",
    save_dir="qiepian",
    patch_size=128,
    stride=64,
    mat_key='input'
):
    """
    mat_path: .mat 文件路径（默认使用仓库根目录下的 matdata.mat）
    save_dir: 切片保存的文件夹（默认在仓库内生成 qiepian/）
    patch_size: 切片大小 (128x128 通常比较适合显存)
    stride: 滑动步长 (越小重叠越多，数据量越大)
    mat_key: .mat 文件中存储数据的变量名，需要你自己检查
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Loading data from {mat_path}...")
    
    # 1. 读取数据
    # 如果是 .npy 直接用 np.load(mat_path)
    try:
        mat_data = sio.loadmat(mat_path)
        # 注意：你需要确认 .mat 里的 key 是什么，常用的是 'd', 'data', 'seismic'
        # 可以用 print(mat_data.keys()) 查看
        full_seismic = mat_data[mat_key] 
    except Exception as e:
        print(f"Error loading .mat: {e}")
        print("Trying to load as numpy array...")
        full_seismic = np.load(mat_path)

    # 确保数据是 float32
    full_seismic = full_seismic.astype(np.float32)
    h, w = full_seismic.shape
    print(f"Original Data Shape: {h} (Time) x {w} (Trace)")

    # 2. 归一化 (Normalization) -> 非常重要！必须归一化到 [-1, 1]
    # 方法：除以绝对值的最大值
    max_val = np.max(np.abs(full_seismic))
    full_seismic = full_seismic / (max_val + 1e-8)
    
    print(f"Data normalized to range [{full_seismic.min():.3f}, {full_seismic.max():.3f}]")

    # 3. 切片 (Patching)
    patches = []
    count = 0
    
    # 双重循环进行滑动窗口切分
    # 只有当原始尺寸大于 patch_size 时才切分
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            # 提取切片
            patch = full_seismic[r : r + patch_size, c : c + patch_size]
            
            # 简单的数据筛选：如果切片全是0（比如静音段），可以丢弃，避免模型学到无效信息
            if np.std(patch) < 1e-5:
                continue
                
            # 增加一个维度变为 (1, H, W) 适配 PyTorch 格式 (Channel, Height, Width)
            # 如果用 CoPaint 原版加载器，可能后续需要 repeat 成 3 通道，这里先存单通道
            patches.append(patch[np.newaxis, :, :])
            count += 1

    # 转换为 numpy 数组
    patches_np = np.array(patches) # Shape: (N, 1, 128, 128)
    
    print(f"Generated {len(patches_np)} patches.")
    
    # 4. 保存
    save_name = os.path.join(save_dir, "marmousi_patches.npy")
    np.save(save_name, patches_np)
    print(f"Saved patches to {save_name}")
    
    # 5. 可视化检查 (可选)
    # 随机画 5 张切片看看对不对
    plt.figure(figsize=(15, 3))
    indices = np.random.choice(len(patches_np), 5, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i+1)
        # patches_np[idx] is (1, H, W), squeeze to (H, W)
        plt.imshow(patches_np[idx].squeeze(), cmap='seismic', vmin=-1, vmax=1)
        plt.axis('off')
        plt.title(f"Patch {idx}")
    plt.savefig(os.path.join(save_dir, "preview_patches.png"))
    print("Preview saved.")

# ================= 使用示例 =================
# 请修改下面的路径
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice seismic sections into patches.")
    parser.add_argument("--mat_path", type=str, default="matdata.mat")
    parser.add_argument("--save_dir", type=str, default="qiepian")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--mat_key", type=str, default="input")
    args = parser.parse_args()

    if not os.path.exists(args.mat_path):
        raise FileNotFoundError(f"Could not find seismic file at {args.mat_path}")

    slice_seismic_data(
        mat_path=args.mat_path,
        save_dir=args.save_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        mat_key=args.mat_key,
    )
