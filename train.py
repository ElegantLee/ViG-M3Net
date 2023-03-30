#!/usr/bin/python3

import argparse
import os
import yaml
import shutil

from trainer import M3Trainer
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/ViG-M3Net.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    # Save log files for each run
    config_file_name = config['name'] + '.yaml'
    config_path = os.path.join(os.getcwd(), 'Yaml')
    path_save = os.path.join(os.getcwd()) + config['save_root'] + config['run_name']
    print(config_path)
    print(path_save)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    shutil.copy(Path(os.path.join(config_path, config_file_name)).as_posix(), path_save)

    trainer = M3Trainer(config)
    trainer.train()
    # trainer.draw_umap()


###################################
if __name__ == '__main__':
    main()
    # pass