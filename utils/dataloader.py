import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.utils import read_split_data


class My_dataset(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image = Image.open(self.images_path[item])
        if image.mode != 'RGB':
            image = image.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_my_dataset(dataset_root_dir, batch_size, train_transform, test_transform, train_split=0.8):
    train_images_path, train_images_label, test_images_path, test_images_label = read_split_data(dataset_root_dir,
                                                                                                 train_split)
    train_dataset = My_dataset(train_images_path, train_images_label, train_transform)
    test_dataset = My_dataset(test_images_path, test_images_label, test_transform)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter


def load_exist_dataset(dataset_root_dir, batch_size, train_transform, test_transform):
    train_dataset = torchvision.datasets.CIFAR10(root=dataset_root_dir,
                                                 train=True,
                                                 download=True,
                                                 transform=train_transform
                                                 )
    test_dataset = torchvision.datasets.CIFAR10(root=dataset_root_dir,
                                                train=False,
                                                download=True,
                                                transform=test_transform
                                                )
    train_iter = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter
