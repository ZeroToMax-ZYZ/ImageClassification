from typing import Tuple
import os
# 避免albumentations更新警告
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_imagenet_transforms(
    input_size=224,
    val_resize_short=256,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[A.Compose, A.Compose]:
    """
    构建 ImageNet-1K 常用增强策略。
    """
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                height=input_size,
                width=input_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333333333),
                interpolation=1,   # cv2.INTER_LINEAR
                p=1.0,
            ),

            # 水平翻转
            A.HorizontalFlip(p=0.5),

            # 轻微几何扰动
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.10,
                rotate_limit=10,
                border_mode=0,     # cv2.BORDER_CONSTANT
                value=0,
                interpolation=1,   # cv2.INTER_LINEAR
                p=0.3,
            ),

            # 颜色与光照扰动
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.8,
            ),

            # 灰度化
            A.ToGray(p=0.05),

            # 轻微模糊/噪声
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
                ],
                p=0.15,
            ),

            # 随机擦除
            A.CoarseDropout(
                max_holes=1,
                max_height=int(0.25 * input_size),
                max_width=int(0.25 * input_size),
                min_holes=1,
                min_height=int(0.10 * input_size),
                min_width=int(0.10 * input_size),
                fill_value=0,
                p=0.25,
            ),

            # 标准归一化 + ToTensor
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    # -------- val --------
    # 经典评测管线：短边缩放到 256 -> 中心裁剪 224
    val_transform = A.Compose(
        [
            A.SmallestMaxSize(
                max_size=val_resize_short,
                interpolation=1,  # cv2.INTER_LINEAR
                p=1.0,
            ),
            A.CenterCrop(height=input_size, width=input_size, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform

if __name__ == "__main__":
    train_transform, val_transform = build_imagenet_transforms()
