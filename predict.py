import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model.model import get_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                                        )

    # load image
    image_path = "dataset/PetImages/Cat/100.jpg"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    image = Image.open(image_path)
    plt.imshow(image)
    # [3, 224, 224]
    image = data_transform(image)
    # expand batch dimension
    # [1, 3, 224, 224]
    image = torch.unsqueeze(image, dim=0)

    # read index2class
    json_path = 'index2class.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as fp:
        index2class = json.load(fp)

    # create model
    model = get_model().to(device)
    # load model weights
    model_weight_path = "model_data/model_2023_11_04_12_47_50/epoch040-train_loss0.201-train_acc0.910-test_loss0.139-test_loss0.947.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(image.to(device))).cpu().item()
        predict_class = int(output > 0.5)
        predict_prob = abs(0.5 - output) / 0.5

    title = "class: {}   prob: {:.3}".format(index2class[str(predict_class)], predict_prob)
    plt.title(title)
    plt.show()
