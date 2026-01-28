from nets.AlexNet import AlexNet

def build_model(cfg):
    model_name = cfg["model_name"]

    if model_name == "AlexNet":
        model = AlexNet(input_size=cfg["input_size"], num_classes=100)

    else:
        raise ValueError(f"❗Unsupported model name: {model_name}")
    
    return model