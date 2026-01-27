import torch
from torch.utils.data import DataLoader, Subset

from dataset.build_dataset import build_dataset
from utils.optim_lr_factory import build_optimizer, build_lr_scheduler

from icecream import ic
import os


def base_config():
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_path": r'D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train',
        "val_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\val",
        # test model 
        "debug_mode": 0.1, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
        "input_size": 224,
        "batch_size": 256,
        "num_workers": 8,
        "persistent_workers": True, # 进程持久化,针对win平台
        "optimizer": {
            "type": "SGD",
            "lr": 0.01,
            "lr_scheduler": {
                "type": "StepLR",
                "step_size": 30,
                "gamma": 0.1,
            },
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        # "optimizer": {
        #     "type": "Adam",
        #     "lr": 0.001,
        #     "lr_scheduler": {
        #         "type": "CosineAnnealingLR",
        #         "T_max": 100,
        #         "eta_min": 1e-6,
        #     },
        #     "weight_decay": 1e-4,
        # }
    
    }
    return config

if __name__ == "__main__":
    if os.name == 'nt':  # 'nt'代表Windows系统
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    cfg = base_config()
    train_loader, val_loader = build_dataset(cfg)

    optimizer = build_optimizer(model=torch.nn.Linear(10, 2), cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)



