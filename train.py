import torch
from torch.utils.data import DataLoader, Subset

from nets.build_model import build_model
from dataset.build_dataset import build_dataset
from utils.optim_lr_factory import build_optimizer, build_lr_scheduler, build_loss_fn
from utils.fit_one_epoch import fit_one_epoch
from utils.logger import save_logger, save_config

from icecream import ic
import os
import time

def base_config():
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "exp_name": "test1",
        "model_name": "AlexNet",
        "save_interval": 5,
        "train_path": r'D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train',
        "val_path": r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\val",
        # test model 
        "debug_mode": 0.1, # 当debug_mode为None时,表示正常模式; 否则为debug模式,使用部分数据训练
        "input_size": 224,
        "batch_size": 128,
        "num_workers": 8,
        "persistent_workers": True, # 进程持久化,针对win平台
        "epochs": 100,
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
        "loss_fn": "CrossEntropyLoss",
        # "optimizer": {
        #     "type": "Adam",
        #     "lr": 0.001,
        #     "lr_scheduler": {
        #         "type": "CosineAnnealingLR",
        #         "T_max": epochs,
        #         "eta_min": 1e-6,
        #     },
        #     "weight_decay": 1e-4,
        # }
    
    }

    config["exp_name"] += str("_" + exp_time)
    return config

def train():
    cfg = base_config()
    save_config(cfg)

    model = build_model(cfg).to(cfg["device"])
    train_loader, val_loader = build_dataset(cfg)

    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    loss_fn = build_loss_fn(cfg)

    for epoch in range(cfg["epochs"]):
        metrics = fit_one_epoch(
            epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler
        )
        # save logs and model
        save_logger(model, metrics, cfg)
        

if __name__ == "__main__":
    if os.name == 'nt':  # 'nt'代表Windows系统
        torch.multiprocessing.set_start_method('spawn', force=True)
    train()
    

        
        







