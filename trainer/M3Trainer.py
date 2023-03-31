#!/usr/bin/python3
import time
from pathlib import Path
import torch.nn
from Model.vig import *
import pandas as pd
from torch.utils.data import DataLoader
from .loss import TripletLoss
from .utils import LambdaLR
from .datasets import ImageDataset, ValDataset
from .utils import Resize, smooothing_loss, ToTensor, tensor2image
from .logger import Logger, TerminalLogger
from .reg import Reg
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
import os
from tqdm import tqdm
from Model.resnet import ResNet
from .pytorch_tcr import Transformation
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'Model')


class M3Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.metrics = {'epoch': [], 'MAE': [], 'PSNR': [], 'SSIM': []}
        self.path_save = os.path.join(os.getcwd()) + self.config['save_root'] + self.config['run_name']
        self.start_epoch_continue = 0
        self.end_epoch_continue = 0

        # ======================= def networks =======================
        """ Generator """
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['g_lr'], betas=(0.5, 0.999))
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(config['n_epochs'], config['epoch'], config['decay_epoch']).step
        )

        """ Regist """
        self.R_A = Reg(config['input_nc'], config['output_nc']).cuda()
        self.spatial_transform = Transformer_2D().cuda()

        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['r_lr'], betas=(0.5, 0.999))
        self.lr_scheduler_R_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_R_A, lr_lambda=LambdaLR(config['n_epochs'], config['epoch'], config['decay_epoch']).step
        )

        """ Deep Metric Learning """
        self.netML = ResNet(opt=config).cuda()
        self.optimizerML = torch.optim.Adam(self.netML.parameters(), lr=config['ml_lr'], betas=(0.5, 0.999))
        self.lr_scheduler_ML = torch.optim.lr_scheduler.LambdaLR(
            self.optimizerML, lr_lambda=LambdaLR(config['n_epochs'], config['epoch'], config['decay_epoch']).step
        )
        # Metric Learning Loss
        self.triplet_ = TripletLoss(config['margin'], config['alpha']).cuda()

        # Losses
        self.MSE_loss = torch.nn.MSELoss(reduction='mean')
        self.L1_loss = torch.nn.L1Loss(reduction='mean')

        # Inputs & targets memory allocation
        self.Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor

        # ======================= Dataset loader -- start =======================
        transforms_1 = [
            # ToPILImage(),
            # RandomAffine(degrees=3, translate=[0.05, 0.05], scale=[0.95, 1.05], fillcolor=-1),
            # degrees=2,translate=[0.05, 0.05],scale=[0.9, 1.1]
            ToTensor(),
            # transforms.Normalize((0.5), (0.5)),
            Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [
            # ToPILImage(),
            # RandomAffine(degrees=3, translate=[0.05, 0.05], scale=[0.95, 1.05], fillcolor=-1),
            ToTensor(),
            # transforms.Normalize((0.5), (0.5)),
            Resize(size_tuple=(config['size'], config['size']))]

        # dataloader for paired data
        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        # dataloader for unpaired data
        # self.dataloader_un = DataLoader(
        #     ImageDataset(config['dataroot_un'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
        #     batch_size=config['batchSize_un'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [
            ToTensor(),
            # transforms.Normalize((0.5), (0.5)),
            Resize(size_tuple=(config['size'], config['size']))]

        self.val_G_transforms = [
            ToTensor(),
            Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['val_batchSize'], shuffle=False, num_workers=config['n_cpu'])
        # ======================= Dataset loader -- end =======================

        # ======================= Logger -- start =======================
        # terminal logger
        self.t_logger = TerminalLogger(config, len(self.dataloader), self.start_epoch_continue, self.end_epoch_continue)
        # tensorboard
        run_name = '-'.join(config['run_name'].split('/'))
        log_dir = os.getcwd() + '/trainer/log/'
        self.logger = Logger(log_root=log_dir, name="{}-{}".format(run_name, config['dataset']))
        counter = 0
        i = 1
        for k, v in config.items():
            self.logger.add_text('configuration_%d' % counter, "{}: {}".format(k, v))
            if i % 10 == 0:
                counter += 1
            i += 1
        # ======================= Logger -- end =======================

    @staticmethod
    def set_requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def test(self):
        # define the path
        result_path = Path(os.getcwd()).as_posix() + self.config['image_save'] + self.config['run_name']
        images_path = Path(os.path.join(result_path, 'images')).as_posix()
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        path_save = os.path.join(os.getcwd()) + self.config['save_root'] + self.config['run_name']
        self.netG_A2B.load_state_dict(
            torch.load(Path(os.path.join(path_save, 'netG_A2B.pth')).as_posix()))

        with torch.no_grad():
            MAE = 0
            PSNR = 0
            SSIM = 0
            num = 0
            metrics = {'MAE': [], 'PSNR': [], 'SSIM': []}
            for i, batch in enumerate(tqdm(self.val_data)):
                real_A = batch['A_img'].cuda()
                real_B = batch['B_img'].cuda().detach().cpu().numpy().squeeze()
                fake_B = self.netG_A2B(real_A)  # .detach().cpu().numpy().squeeze()
                fake_B = fake_B.detach().cpu().numpy().squeeze()
                mae = self.MAE(real_B, fake_B)
                psnr = measure.compare_psnr(real_B, fake_B)
                if self.config['input_nc'] == 1:
                    ssim = measure.compare_ssim(real_B, fake_B)
                else:
                    ssim = measure.compare_ssim(np.transpose(real_B, (1, 2, 0)), np.transpose(fake_B, (1, 2, 0)),
                                                multichannel=True)

                MAE += mae
                PSNR += psnr
                SSIM += ssim
                num += 1

                real_A = ((real_A.detach().cpu().numpy().squeeze() + 1) / 2) * 255
                real_B = ((real_B + 1) / 2) * 255
                fake_B = 255 * ((fake_B + 1) / 2)

                if self.config['input_nc'] != 1:
                    fake_B = np.transpose(fake_B, (1, 2, 0))
                    real_B = np.transpose(real_B, (1, 2, 0))
                    real_A = np.transpose(real_A, (1, 2, 0))

                cv2.imwrite(result_path + '/images/' + str(num) + '_real_A.png', real_A)
                cv2.imwrite(result_path + '/images/' + str(num) + '_real_B.png', real_B)
                cv2.imwrite(result_path + '/images/' + str(num) + '_fake_B.png', fake_B)
                metrics['MAE'].append(mae)
                metrics['PSNR'].append(psnr)
                metrics['SSIM'].append(ssim)

            MAE = MAE / num
            PSNR = PSNR / num
            SSIM = SSIM / num
            metrics['MAE'].append(MAE)
            metrics['PSNR'].append(PSNR)
            metrics['SSIM'].append(SSIM)

            print('MAE:%.3f, PSNR:%.3f, SSIM:%.3f' % (MAE, PSNR, SSIM))
            df_metrics = pd.DataFrame.from_dict(metrics)
            metrics_path = Path(result_path + 'metrics.csv').as_posix()
            # print('metrics_path: ', metrics_path)
            df_metrics.to_csv(metrics_path, encoding='utf-8')

    def MAE(self, fake, real):
        """ graysale """
        # x, y = np.where(real != -1)  # coordinate of target points
        # # points = len(x)  #num of target points
        # mae = np.abs(fake[x, y] - real[x, y]).mean()
        #
        # return mae / 2
        """ RGB """
        mae = np.abs(fake - real).mean()
        return mae