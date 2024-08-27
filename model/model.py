import torch
import torchvision
import torch.nn as nn


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()

    def forward(self):
        pass


def get_model():
    model = nn.Sequential(torchvision.models.resnet18(),
                          nn.Linear(1000, 1),
                          nn.Sigmoid()
                          )
    return model
