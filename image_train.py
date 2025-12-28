"""
Train a diffusion model on Seismic Images (Numpy format).
"""
import sys
sys.path.append("/data/ddim/CoPaint-master")
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from guided_diffusion import dist_util, logger
# from guided_diffusion.image_datasets import load_data  <-- 删除或注释掉这一行
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# --- 新增：定义你的加载函数 ---
def load_seismic_data(data_path, batch_size, image_size):
    print(f"Loading seismic patches from {data_path}...")
    patches = np.load(data_path)
    
    # 增加通道维度 (N, 128, 128) -> (N, 1, 128, 128)
    if patches.ndim == 3:
        patches = patches[:, np.newaxis, :, :]
        
    tensor_data = torch.from_numpy(patches).float()
    
    # 注意：这里第二个参数传空字典，因为扩散模型训练不需要标签
    dataset = TensorDataset(tensor_data, torch.zeros(len(tensor_data))) 
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    while True:
        # yield 返回 batch 数据，TrainLoop 需要字典格式
        for batch in loader:
            # batch[0] 是图像数据
            yield batch[0], {} 
# -----------------------------
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    
    # -------- 核心修改开始 --------
    # 1. 将 args 转换为字典
    args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    
    # 2. 【强制】覆盖通道数参数，不管命令行传了什么，这里直接写死为 1
    print("Force overriding channels to 1 for Seismic Data...")
    args_dict['in_channels'] = 1
    args_dict['out_channels'] = 1
    
    # 3. 使用修改后的字典创建模型
    model, diffusion = create_model_and_diffusion(**args_dict)
    # -------- 核心修改结束 --------

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # ... (数据加载部分保持你之前的修改)
    data = load_seismic_data(
        data_path=args.data_dir, # 或者你硬编码的路径
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1, 
        ema_rate="0.9999",
        log_interval=100,      # 建议改大一点，避免刷屏
        save_interval=5000,    # 每5000步保存一次
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    # 这里会加载模型默认参数 (num_channels, image_size 等)
    model_defaults = model_and_diffusion_defaults()
    model_defaults['in_channels'] = 1   # 强制设为 1
    model_defaults['out_channels'] = 1  # 强制设为 1
    defaults.update(model_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
