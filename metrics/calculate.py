# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: Reg-GAN-main
# @File  : calculate
# @Author: super
# @Date  : 2021/11/28
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import tqdm
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from piq import fsim, multi_scale_ssim, FID
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn.functional as F

class Resize():
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])
        tensor = tensor.squeeze(0)

        return tensor  # 1, 64, 128, 128


class ToTensor():
    def __call__(self, tensor):
        """
        tensor: H W C
        target: C H W
        """
        if len(tensor.shape) == 2:
            tensor = np.expand_dims(tensor, 0)
            tensor = np.array(tensor)
        elif len(tensor.shape) == 3:
            tensor = np.array(tensor)
            return torch.from_numpy(tensor.transpose(2, 0, 1))  # C H W
        return torch.from_numpy(tensor)  # C H W

def cal_MSSSIM(real_img, fake_img):
    ms_ssim_score = multi_scale_ssim(real_img, fake_img, data_range=1.)
    return ms_ssim_score.item()


def cal_FSIM(real_img, fake_img):
    fsim_score = fsim(real_img, fake_img, data_range=1., chromatic=False)
    return fsim_score.item()


def cal_MMD(real_imgs, fake_imgs, feat_model, MMD):
    # resnet34 = models.resnet34(pretrained=True)
    # for param in resnet34.parameters():
    #     param.requires_grad = False
    # resnet34.eval()
    # resnet34 = resnet34.cuda()
    # # weight = resnet34.state_dict()
    #
    # MMD = MMD_loss().cuda()
    # resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    real_imgs_feat = feat_model(real_imgs)
    fake_imgs_feat = feat_model(fake_imgs)
    mmd_score = MMD(real_imgs_feat, fake_imgs_feat).item()
    return mmd_score


def cal_FID(real_img_dataloader, fake_img_dataloader):
    FID_compute = FID()
    real_img_feat = FID_compute.compute_feats(real_img_dataloader)
    fake_img_feat = FID_compute.compute_feats(fake_img_dataloader)
    fid_score = FID_compute.compute_metric(real_img_feat, fake_img_feat)
    return fid_score.item()


def GAN_train():

    pass


def GAN_test():
    pass


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class ImageDataset(Dataset):
    def __init__(self, path, direction, transform_1, transform_2):
        # root = os.path.join(os.getcwd(), path)
        self.real_imgs_path = []
        self.fake_imgs_path = []
        for img_name in os.listdir(path):
            info = img_name.split('_')
            if direction == 'AtoB':
                if 'B' in info[2]:
                    if 'real' in info[1]:
                        real_img_path = os.path.join(path, img_name)
                        self.real_imgs_path.append(real_img_path)
                    elif 'fake' in info[1]:
                        fake_img_path = os.path.join(path, img_name)
                        self.fake_imgs_path.append(fake_img_path)
            elif direction == 'BtoA':
                if 'A' in info[2]:
                    if 'real' in info[2]:
                        real_img_path = os.path.join(path, img_name)
                        self.real_imgs_path.append(real_img_path)
                    elif 'fake' in info[2]:
                        fake_img_path = os.path.join(path, img_name)
                        self.fake_imgs_path.append(fake_img_path)

        self.real_imgs_size = len(self.real_imgs_path)
        self.fake_imgs_size = len(self.real_imgs_path)
        self.transform_1 = transforms.Compose(transform_1)
        self.transform_2 = transforms.Compose(transform_2)

    def __getitem__(self, index):
        real_img_path = self.real_imgs_path[index % self.real_imgs_size]
        fake_img_path = self.fake_imgs_path[index % self.fake_imgs_size]

        real_img = cv2.imread(real_img_path, 0)
        real_img = (real_img / float(np.max(real_img))).astype(np.float32)
        real_img = self.transform_1(real_img)

        fake_img = cv2.imread(fake_img_path, 0)
        fake_img = (fake_img / np.max(fake_img)).astype(np.float32)
        fake_img = self.transform_2(fake_img)
        return {'real_img': real_img, 'fake_img': fake_img}

    def __len__(self):
        return self.real_imgs_size


class FakeImageDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


if __name__ == '__main__':
    img = torch.rand(1, 1, 256, 256).cuda()
    batch_size = 1
    metrics = {'num_iter': [], 'MS-SSIM': [], 'FSIM': []}
    # transform
    transform_1 = [
        ToTensor(),
        # transforms.Normalize((0.5), (0.5)),
        Resize(size_tuple=(256, 256))
    ]

    transform_2 = [
        ToTensor(),
        Resize(size_tuple=(256, 256))
    ]
    supp_1 = len(os.getcwd().split('\\')[-2]) - 1
    imgs_path = os.path.join(os.getcwd()[:-supp_1], 'results\\MMNet\\IXI\\batch-16\\01_ResNet18\\images')
    result_path = imgs_path[:-6]
    direction = 'AtoB'
    metric_data = DataLoader(ImageDataset(imgs_path, direction, transform_1, transform_2), batch_size=batch_size, shuffle=False, num_workers=3)

    MS_SSIM = 0.
    FSIM = 0.
    MMD = 0.

    num_iter = 0
    for i, batch in enumerate(tqdm.tqdm(metric_data)):
        real_img = batch['real_img'].cuda()
        fake_img = batch['fake_img'].cuda()
        # mmd_socre = cal_MMD(real_img, fake_img, resnet34, MMD_model)
        ms_ssim_score = cal_MSSSIM(real_img, fake_img)
        fsim_score = cal_FSIM(real_img, fake_img)
        MS_SSIM += ms_ssim_score
        FSIM += fsim_score
        # MMD += mmd_socre
        metrics['MS-SSIM'].append(ms_ssim_score)
        metrics['FSIM'].append(fsim_score)
        # metrics['MMD'].append(mmd_socre)
        num_iter += 1
        metrics['num_iter'].append(num_iter)



    avg_MS_SSIM = MS_SSIM / len(metric_data)
    avg_FSIM = FSIM / len(metric_data)
    avg_MMD = MMD / len(metric_data)


    metrics['MS-SSIM'].append(avg_MS_SSIM)
    metrics['FSIM'].append(avg_FSIM)
    metrics['num_iter'].append(num_iter+1)

    df_metrics = pd.DataFrame.from_dict(metrics)
    metrics_path = Path(result_path + '/local_metrics.csv').as_posix()
    # print('metrics_path: ', metrics_path)
    df_metrics.to_csv(metrics_path, encoding='utf-8')

    print('avg_MSSIM: %.3f, avg_FSIM: %.3f' % (avg_MS_SSIM, avg_FSIM))

