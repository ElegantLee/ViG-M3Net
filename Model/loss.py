# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: Reg-GAN-main
# @File  : loss
# @Author: super
# @Date  : 2021/10/31
import torch


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss