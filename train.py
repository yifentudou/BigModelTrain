import json
import torch
from torchvision import transforms
from model.model import get_model
from trainer.trainer import get_loss_fn, train
from utils.dataloader import load_my_dataset
from utils.utils import show_config, LossHistory

if __name__ == '__main__':
    # 指定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 指定配置文件路径
    config_dir = "./config/config.json"
    # 加载配置文件
    with open(config_dir, 'r') as fp:
        config = json.load(fp)
    dataset_config = config["dataset"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    train_config = config["train"]
    # 显示配置
    show_config(**{**dataset_config, **model_config, **optimizer_config, **train_config})
    # 创建日志记录器
    loss_history = LossHistory(log_dir=train_config["log_dir"], config_dir=config_dir,
                               model_save_dir=train_config["model_save_dir"])

    # 加载数据迭代器
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                                         )
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                                        )
    train_iter, test_iter = load_my_dataset(dataset_root_dir=dataset_config["root_dir"],
                                            batch_size=dataset_config["batch_size"],
                                            train_transform=train_transform,
                                            test_transform=test_transform,
                                            train_split=dataset_config["train_split"]
                                            )

    # 创建模型
    model = get_model().to(device)

    # 优化器
    type = optimizer_config.pop("type")
    optimizer = {"Adam": torch.optim.Adam(model.parameters(), **optimizer_config),
                 "SGD": torch.optim.SGD(model.parameters(), **optimizer_config)}[type]

    # 损失函数
    loss_fn = get_loss_fn()

    # 训练
    train(train_config["num_epochs"], len(train_iter), len(test_iter), train_iter, test_iter, model,
          optimizer, loss_fn, loss_history, train_config["model_save_period"], device)
