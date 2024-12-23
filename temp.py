import os
from PIL import Image
import torch
import random
import datetime
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import build_model
from data import build_dataset
from utils import get_logger, MetricLogger, SmoothedValue
from args import add_management_args, add_experiment_args, add_bayes_args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(name="BayeSeg", root=self.output_dir)
        self.logger.info(args)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "summary"))
        self.device = torch.device(args.device)

        self.model, self.criterion, self.visualizer = build_model(args)
        self.model.to(self.device)
        self.logger.info(self.model)

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info("number of params:{}".format(n_parameters))

        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
        ]

        self.optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.logger.info("Building training dataset...")
        dataset_train = build_dataset(image_set="train", args=args)
        self.logger.info("Number of training images: {}".format(len(dataset_train)))

        self.logger.info("Building validation dataset...")
        dataset_val = build_dataset(image_set="val", args=args)
        self.logger.info("Number of validation images: {}".format(len(dataset_val)))

        self.train_loader = DataLoader(
            dataset_train,
            args.batch_size,
            True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            dataset_val,
            args.batch_size,
            False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.batch_size = args.batch_size
        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.best_dice = None

    def train_one_epoch(self):
        train_iterator = iter(self.train_loader)
        valid_iterator = iter(self.valid_loader)
        data_dict = next(train_iterator)
        image_data = data_dict["image"][5]
        ori_image_data = data_dict["ori_image"][5]
        self.getminmax(ori_image_data)

        #normalized_image = data_dict["image"][5]
        original_min, original_max, original_mean, original_std = self.getminmax(image_data)

        # self.getminmax(normalized_image)

        # restored_image = normalized_image * (original_max - original_min) + original_min
        # restored_image = normalized_image * original_std + original_mean

        self.save(image_data, "runs/image_data.png")
        self.save(ori_image_data, "runs/ori_image_data.png")
        #self.save(restored_image, "runs/restored_image.png")

    def save(self, image_data, output):
        image_data = image_data.squeeze().numpy()
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))  # 归一化到 [0, 1]
        image_data = (image_data * 255).astype(np.uint8)  # 将数据缩放到 [0, 255] 并转换为无符号整数
        image = Image.fromarray(image_data)
        image.save(output)

    def getminmax(self,image_data):
        original_min = image_data.min().item()  # 记录原始图像的最小值
        original_max = image_data.max().item()  # 记录原始图像的最大值
        original_mean = image_data.mean().item()
        original_std = image_data.std().item()
        print("mean:",original_mean," std:",original_std)
        print("min:",original_min," max:",original_max)

        return original_min, original_max, original_mean, original_std





if __name__ == "__main__":
    parser = argparse.ArgumentParser("BayeSeg training", allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train_one_epoch()
