from nets.AlexNet import AlexNet
from nets.yolov1 import YOLOv1_Classifier

def build_model(cfg):
    model_name = cfg["model_name"]

    if model_name == "AlexNet":
        model = AlexNet(input_size=cfg["input_size"], num_classes=100)

    if model_name == "YOLOv1_backbone":
        model = YOLOv1_Classifier(num_classes=100, ic_debug=False)
    else:
        raise ValueError(f"❗Unsupported model name: {model_name}")
    
    return model