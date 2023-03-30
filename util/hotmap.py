# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project: ViG-M3Net
# @File  : hotmap
# @Author: ElegantLee
# @Date  : 2021/12/3
import glob
import os
from pathlib import Path

import numpy as np
import tqdm
from matplotlib import image as mpimg, pyplot as plt


def get_cyc_hotmap(path, save_path):
    # file_names = glob.glob(path + '*_real_*.png')
    path = Path(path).as_posix()
    save_path = Path(save_path).as_posix()
    '''unpair'''
    file_names = glob.glob(path + '/*_real_B.png')

    num_i=0
    # list_mae = []
    for real_file_name in tqdm.tqdm(file_names):

        num_i=num_i+1
        real_file_name = Path(real_file_name).as_posix()
        file_num_1 = real_file_name.split("/")[-1].split('.')[0].split('_')[0]
        # file_num_2 = real_file_name.split("/")[-1].split('.')[0].split('_')[1]
        A_or_B = real_file_name.split("/")[-1].split('.')[0].split('_')[-1]

        fake_file_name = path + '/' + file_num_1 + '_fake_' + A_or_B + '.png'

        img_real = mpimg.imread(real_file_name)
        # img_fake = mpimg.imread(fake_file_name)
        if len(img_real.shape) < 3:
            img_real = np.expand_dims(mpimg.imread(real_file_name), axis=2)
            img_fake = np.expand_dims(mpimg.imread(fake_file_name), axis=2)
        else:
            # img_real = mpimg.imread(real_file_name)
            img_fake = mpimg.imread(fake_file_name)

        # print(real_file_name)
        # print(fake_file_name)

        img = abs(img_real - img_fake)

        for i in range(len(img[0])):
            for j in range(len(img[1])):
                for k in range(len(img[0][1])):
                    if img[i][j][k] > 0.16:
                        img[i][j][k] = img[i][j][k]
                    else:
                        img[i][j][k] = 0

        lum_img = img[:, :, 0]
        # plt.imshow(lum_img)
        plt.imshow(lum_img, cmap="nipy_spectral")
        # imgplot = plt.imshow(lum_img)
        # imgplot.set_cmap('nipy_spectral')
        # imgplot = plt.imshow(lum_img)
        plt.colorbar()
        # save_hotmap_path = save_path + '/' + real_file_name.split("/")[-1].split('.')[0] + '_hot.png'
        plt.savefig(save_path + '/' + real_file_name.split("/")[-1].split('.')[0] + '_hot.png', bbox_inches='tight', dpi=64)
        plt.clf()

    # print(num_i)

if __name__ == '__main__':
    supp_1 = len(os.getcwd().split('\\')[-1])
    sub_path = 'path'
    images_path = os.path.join(os.getcwd()[:-supp_1], sub_path, 'images')
    hotmap_save_path = os.path.join(os.getcwd()[:-supp_1], sub_path, 'hotmaps')
    if not os.path.exists(hotmap_save_path):
        os.mkdir(hotmap_save_path)
    get_cyc_hotmap(images_path, hotmap_save_path)