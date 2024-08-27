import torch
import thop
from torchsummary import summary

from model.model import get_model


if __name__ == '__main__':
    device = torch.device('cuda')

    model_path = 'model_data/model_2023_11_04_12_47_50/epoch040-train_loss0.201-train_acc0.910-test_loss0.139-test_loss0.947.pth'
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    # model = model.to(device)

    input_size = (1, 3, 224, 224)
    # summary(model, (3, 224, 224))
    input_data = torch.randn(input_size)

    flops, params = thop.profile(model, inputs=(input_data,))
    print(f"FLOPS: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")
