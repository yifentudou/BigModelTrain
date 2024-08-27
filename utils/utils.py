import json
import os
from datetime import datetime
import random

import torch
from matplotlib import pyplot as plt


class LossHistory:
    def __init__(self, log_dir, config_dir, model_save_dir):
        time_str = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, "log_" + str(time_str))
        self.model_dir = os.path.join(model_save_dir, "model_" + str(time_str))
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        os.makedirs(self.log_dir)
        os.makedirs(self.model_dir)

        # 存储此次训练的参数
        with open(os.path.join(self.log_dir, "config.json"), 'w') as f:
            with open('./' + config_dir, 'r') as _f:
                config = json.load(_f)
            json.dump(config, f, indent=4)
        with open(os.path.join(self.model_dir, "config.json"), 'w') as f:
            with open('./' + config_dir, 'r') as _f:
                config = json.load(_f)
            json.dump(config, f, indent=4)

    def append_loss(self, _train_loss, _test_loss, _train_acc, _test_acc):
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)

        self.train_loss.append(_train_loss)
        self.test_loss.append(_test_loss)
        self.train_acc.append(_train_acc)
        self.test_acc.append(_test_acc)

        with open(os.path.join(self.log_dir, "train_loss.txt"), 'a') as f:
            f.write(str(_train_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "test_loss.txt"), 'a') as f:
            f.write(str(_test_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "train_acc.txt"), 'a') as f:
            f.write(str(_train_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "test_acc.txt"), 'a') as f:
            f.write(str(_test_acc))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(1, len(self.train_loss) + 1)

        plt.figure("Loss")
        plt.plot(iters, self.train_loss, 'red', linewidth=2, label='train_loss')
        plt.plot(iters, self.test_loss, 'green', linewidth=2, label='test_loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

        plt.figure("Acc")
        plt.plot(iters, self.train_acc, 'red', linewidth=2, label='train_acc')
        plt.plot(iters, self.test_acc, 'green', linewidth=2, label='test_acc')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
        plt.cla()
        plt.close("all")

    def save_model(self, model, epoch, train_loss, test_loss, train_acc, test_acc):
        torch.save(model.state_dict(),
                   os.path.join(self.model_dir,
                                "epoch%03d-train_loss%.3f-train_acc%.3f-test_loss%.3f-test_loss%.3f.pth" %
                                (epoch, train_loss, train_acc, test_loss, test_acc)
                                )
                   )


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def read_split_data(root_dir, train_split: float = 0.8):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root_dir), "dataset root: {} does not exist.".format(root_dir)

    # 遍历文件夹，一个文件夹对应一个类别
    all_classes = [cla for cla in os.listdir(root_dir)]
    # 排序，保证各平台顺序一致
    all_classes.sort()
    # 生成类别名称以及对应的数字索引
    class2index = dict((k, v) for v, k in enumerate(all_classes))
    json_str = json.dumps(dict((val, key) for key, val in class2index.items()), indent=4)
    with open('index2class.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    # 遍历每个文件夹下的文件
    for cla in all_classes:
        cla_path = os.path.join(root_dir, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root_dir, cla, i) for i in os.listdir(cla_path)]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class2index[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        test_path = random.sample(images, k=int(len(images) * (1. - train_split)))

        for img_path in images:
            if img_path in test_path:  # 如果该路径在采样的验证集样本中则存入验证集
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    return train_images_path, train_images_label, test_images_path, test_images_label
