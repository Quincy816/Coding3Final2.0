import torchvision.models as models
import torch.nn as nn
import torch

def build_model(pretrained=True, fine_tune=True):
    if pretrained:
        print('加载预训练')
    elif not pretrained:
        print('不加载预训练')
    model = models.mobilenet_v2(pretrained=pretrained)

    if fine_tune:
        print('冻结所有层')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('冻结部分')
        for params in model.parameters():
            params.requires_grad = True


    model.fc = nn.Linear(1024, 5)
    return model
