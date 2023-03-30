#!/usr/bin/python3
import time
from pathlib import Path
import torch.nn
from Model.ViG import *
import pandas as pd
from torch.utils.data import DataLoader
from .loss import TripletLoss, pairwise_distances
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
        self.logger = Logger(log_root=log_dir, name="{}-{}".format(run_name, config['data']))
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

    def train(self):
        torch.manual_seed(123)
        time_start = time.time()
        image_plot_interval = 10

        # define the path
        result_path = Path(os.getcwd()).as_posix() + self.config['save_root'] + self.config['run_name']
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # continue training
        start_epoch = self.config['epoch'] + self.start_epoch_continue
        end_epoch = self.config['n_epochs'] + self.end_epoch_continue

        # affine transformation
        tcr = Transformation()
        device = torch.device("cuda")

        # ======================= Training -- start =======================
        for epoch in range(start_epoch, end_epoch):
            interval = 0

            SR_loss_per_epoch = 0.
            SM_loss_per_epoch = 0.
            tcr_loss_per_epoch = 0.
            img_B_loss_per_epoch = 0.
            percp_B_per_epoch = 0.
            d_c_per_epoch = 0.  # distance of centroids
            d_g_per_epoch = 0.  # Euclidean distance of global pairwise points
            mm_loss_per_epoch = 0.
            ml_loss_per_epoch = 0.

            # ===================== epoch -- start =========================
            for i, batch in enumerate(zip(self.dataloader, self.dataloader_un), 0):
                data_sup, data_un = batch[0], batch[1]
                A_img, B_img = data_sup['A_img'], data_sup['B_img']
                A_img_un = data_un['A_img']
                batch_size = len(A_img)
                interval += 1
                # Set model input
                # (1, 3, 256, 256)
                real_A = A_img.cuda()
                real_B = B_img.cuda()

                real_A_un = A_img_un.cuda()

                # Applying our TCR on the Unsupervised data
                random = torch.rand((real_A.shape[0], 4), device=device)
                random_un = torch.rand((real_A_un.shape[0], 4), device=device)

                # ==================== distribution generator ==========================
                self.set_requires_grad(self.netG_A2B, True)
                self.set_requires_grad(self.R_A, True)
                self.set_requires_grad(self.netML, False)
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()

                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B, real_B)
                SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                real_A_transformed = tcr(real_A, random)
                fake_B_transformed = self.netG_A2B(real_A_transformed)
                transformed_fake_B = tcr(fake_B, random)

                fake_B_un = self.netG_A2B(real_A_un)
                real_A_un_transformed = tcr(real_A_un, random_un)
                fake_B_un_transformed = self.netG_A2B(real_A_un_transformed)
                transformed_fake_B_un = tcr(fake_B_un, random_un)
                # correction loss
                SR_loss = 20 * self.L1_loss(SysRegist_A2B, real_B)
                # smoothness loss
                SM_loss = 10 * smooothing_loss(Trans)
                # pL1 loss
                img_B_loss = self.L1_loss(real_B, fake_B) * self.config['lambda_img']
                # manifold affine transformation consistency regularization
                tcr_loss = (self.L1_loss(fake_B_transformed, transformed_fake_B) + self.L1_loss(fake_B_un_transformed,
                                                                                                transformed_fake_B_un)) * \
                           self.config['lambda_tcr']

                ml_real_B_out = self.netML(real_B)
                ml_fake_B_out = self.netML(fake_B)

                # ==================== manifold matching ==========================
                # d_pair
                pd_r = pairwise_distances(ml_real_B_out, ml_real_B_out)
                pd_f = pairwise_distances(ml_fake_B_out, ml_fake_B_out)
                p_dist = torch.dist(pd_r, pd_f, 2) * self.config['lambda_mm']  # matching 2-diameters
                # d_average
                c_dist = torch.dist(ml_real_B_out.mean(0), ml_fake_B_out.mean(0), 2) * self.config[
                    'lambda_mm']  # matching centroids
                # manifold matching loss
                mm_loss = p_dist + c_dist
                # paired deep feature loss
                percp_B_loss = self.MSE_loss(ml_real_B_out, ml_fake_B_out) * self.config['lambda_percp']

                # total_loss
                total_GA2B_loss = mm_loss \
                                  + SR_loss \
                                  + SM_loss \
                                  + img_B_loss \
                                  + percp_B_loss \
                                  + tcr_loss \

                total_GA2B_loss.backward()
                self.optimizer_R_A.step()
                self.optimizer_G.step()

                # ========================= distribution discriminator ============================
                self.set_requires_grad(self.netG_A2B, False)
                self.set_requires_grad(self.R_A, False)
                self.set_requires_grad(self.netML, True)
                self.optimizerML.zero_grad()

                fake_B = self.netG_A2B(real_A)
                ml_real_B_out = self.netML(real_B)
                ml_fake_B_out = self.netML(fake_B)

                r1 = torch.randperm(batch_size)
                r2 = torch.randperm(batch_size)
                ml_real_B_out_shuffle = ml_real_B_out[r1[:, None]].view(ml_real_B_out.shape[0],
                                                                        ml_real_B_out.shape[-1])
                ml_fake_B_out_shuffle = ml_fake_B_out[r2[:, None]].view(ml_fake_B_out.shape[0],
                                                                        ml_fake_B_out.shape[-1])
                # anchor, positive, negative
                ml_loss = self.triplet_(ml_real_B_out, ml_real_B_out_shuffle, ml_fake_B_out_shuffle)
                ml_loss.backward()
                self.optimizerML.step()

                # ======================== training -- end ==============================
                if interval % image_plot_interval == 0:
                    do_plot = True
                else:
                    do_plot = True

                # terminal log
                self.t_logger.log({
                    'SR_loss': SR_loss,
                    'SM_loss': SM_loss,
                    'mm_loss': mm_loss,
                    'img_loss': img_B_loss,
                    'tcr_loss': tcr_loss,
                    'ml_loss': ml_loss,
                })

                # ======================== tensorboard-image ==============================
                if do_plot:
                    # img_index = random.randint(0, self.config['batchSize'] - 1)
                    real_A_img = tensor2image(real_A.data).transpose(2, 0, 1)
                    real_B_img = tensor2image(real_B.data).transpose(2, 0, 1)
                    fake_B_img = tensor2image(fake_B.data).transpose(2, 0, 1)
                    real_A_transformed_img = tensor2image(real_A_transformed.data).transpose(2, 0, 1)
                    fake_B_transformed_img = tensor2image(fake_B_transformed.data).transpose(2, 0, 1)
                    transformed_fake_B_img = tensor2image(transformed_fake_B.data).transpose(2, 0, 1)
                    R_B_img = tensor2image(SysRegist_A2B.data).transpose(2, 0, 1)
                    # re_real_B_img = tensor2image(re_real_B.data).transpose(2, 0, 1)
                    # re_fake_B_img = tensor2image(re_fake_B.data).transpose(2, 0, 1)

                    self.logger.add_image('real_A', real_A_img, epoch + 1)
                    self.logger.add_image('real_A_transformed', real_A_transformed_img, epoch + 1)
                    self.logger.add_image('fake_B', fake_B_img, epoch + 1)
                    self.logger.add_image('transformed_fake_B', transformed_fake_B_img, epoch + 1)
                    self.logger.add_image('fake_B_transformed', fake_B_transformed_img, epoch + 1)
                    self.logger.add_image('real_B', real_B_img, epoch + 1)
                    self.logger.add_image('regist_B', R_B_img, epoch + 1)
                    # self.logger.add_image('re_real_B', re_real_B_img, epoch + 1)
                    # self.logger.add_image('re_fake_B', re_fake_B_img, epoch + 1)

                # regist loss
                SR_loss_per_epoch += SR_loss.item()
                SM_loss_per_epoch += SM_loss.item()

                # content loss
                img_B_loss_per_epoch += img_B_loss.item()
                percp_B_per_epoch += percp_B_loss.item()

                # tcr loss
                tcr_loss_per_epoch += tcr_loss.item()

                # manifold matching loss
                mm_loss_per_epoch += mm_loss.item()
                ml_loss_per_epoch += ml_loss.item()
                d_g_per_epoch += p_dist.item()
                d_c_per_epoch += c_dist.item()

            # ===================== epoch -- end =========================

            # ===================== tensorboard-scalar =========================
            SR_loss_plot = SR_loss_per_epoch / len(self.dataloader)
            SM_loss_plot = SM_loss_per_epoch / len(self.dataloader)
            img_B_loss_plot = img_B_loss_per_epoch / len(self.dataloader)
            percp_B_loss_plot = percp_B_per_epoch / len(self.dataloader)
            tcr_loss_plot = tcr_loss_per_epoch / len(self.dataloader)
            mm_loss_plot = mm_loss_per_epoch / len(self.dataloader)
            d_g_plot = d_g_per_epoch / len(self.dataloader)
            d_c_plot = d_c_per_epoch / len(self.dataloader)
            ml_loss_plot = ml_loss_per_epoch / len(self.dataloader)
            # =====================使用tensorboard======================
            """ regist loss """
            self.logger.add_scalar('SR_loss', SR_loss_plot, epoch + 1)
            self.logger.add_scalar('SM_loss', SM_loss_plot, epoch + 1)

            """ content loss """
            self.logger.add_scalar('img_B_loss', img_B_loss_plot, epoch + 1)
            self.logger.add_scalar('percp_B_loss', percp_B_loss_plot, epoch + 1)

            self.logger.add_scalar('tcr_loss', tcr_loss_plot, epoch + 1)

            """ manifold matching loss """
            self.logger.add_scalar('d_centroids', d_c_plot, epoch + 1)
            self.logger.add_scalar('d_diameters(2)', d_g_plot, epoch + 1)
            self.logger.add_scalar('mm_loss', mm_loss_plot, epoch + 1)
            self.logger.add_scalar('ml_loss', ml_loss_plot, epoch + 1)

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_R_A.step()
            self.lr_scheduler_ML.step()

            # ===================== val =========================
            with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0

                for i, batch in enumerate(tqdm(self.val_data)):
                    real_A = batch['A_img'].cuda()
                    real_B = batch['B_img'].cuda().detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()

                    mae = self.MAE(real_B, fake_B)
                    psnr = measure.compare_psnr(real_B, fake_B)
                    if self.config['data'] == 'BrainTs2015':
                        ssim = measure.compare_ssim(real_B, fake_B)
                    else:
                        ssim = measure.compare_ssim(np.transpose(real_B, (1, 2, 0)), np.transpose(fake_B, (1, 2, 0)),
                                                    multichannel=True)

                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim
                    num += 1

                MAE = MAE / num
                PSNR = PSNR / num
                SSIM = SSIM / num

                if epoch >= 29:
                    torch.save(self.netG_A2B.state_dict(),
                               Path(os.path.join(self.path_save, 'epoch%d_' % (epoch + 1) + 'netG_A2B.pth')).as_posix())
                print('MAE:%.5f, PSNR:%.5f, SSIM:%.5f' % (MAE, PSNR, SSIM))
                self.logger.add_scalar('MAE', MAE, epoch + 1)
                self.logger.add_scalar('PSNR', PSNR, epoch + 1)
                self.logger.add_scalar('SSIM', SSIM, epoch + 1)

        time_train = (time.time() - time_start) / 3600.0
        print('----------------------------------')
        print('train is over, costs %.2f h' % time_train)

    def test(self, ):
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

    def save_deformation(self, defms, root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max, x_min = dir_x.max(), dir_x.min()
        y_max, y_min = dir_y.max(), dir_y.min()
        dir_x = ((dir_x - x_min) / (x_max - x_min)) * 255
        dir_y = ((dir_y - y_min) / (y_max - y_min)) * 255
        # print (x_max,x_min)
        # print (y_max,y_min)
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_x[tans_x <= 150] = 0
        # print (tans_x.shape,tans_x.max(),tans_x.min())
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_y[tans_y <= 150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy)