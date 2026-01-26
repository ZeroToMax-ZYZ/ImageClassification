from dataset.augment import build_imagenet_transforms
from dataset.dataset_img100 import ImageNet100

from icecream import ic

train_transform, val_transform = build_imagenet_transforms()

train_path = r'D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\train'
val_path = r"D:\1AAAAAstudy\python_base\pytorch\all_dataset\image_classification\ImageNet\ImageNet100\val"

train_dataset = ImageNet100(train_path, transform=train_transform)

label_index = train_dataset._get_index()
val_dataset = ImageNet100(val_path, label_index, transform=val_transform)

ic(train_dataset[0])