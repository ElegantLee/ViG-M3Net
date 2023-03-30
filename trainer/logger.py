import datetime
import os
import sys
import time

# from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def flatten_dict(dic):
    flattned = dict()

    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten(k, v)
                else:
                    _flatten(prefix + '/%s' % k, v)
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[prefix + '/%s' % k] = v

    _flatten(None, dic)
    return flattned


class Logger(SummaryWriter):
    def __init__(self, log_root='', name='', logger_name=''):
        if logger_name == '':
            # date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            # self.log_name = '{}_{}'.format(name, date)
            log_dir = os.path.join(log_root, name)
            super(Logger, self).__init__(log_dir, flush_secs=1)
        else:
            self.log_name = 'test_' + logger_name
            log_dir = os.path.join(log_root, name)
            super(Logger, self).__init__(log_dir, flush_secs=1)


class TerminalLogger():
    def __init__(self, config, batches_epoch, start_epoch_continue, end_epoch_continue):
        self.n_epochs = config['n_epochs'] + end_epoch_continue
        self.batches_epoch = batches_epoch
        self.epoch = 1 + start_epoch_continue
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}

        self.config = config
        self.env_name = config['name']

    def log(self, losses=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                self.losses[loss_name] = 0.0
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
