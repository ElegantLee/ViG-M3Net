import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from gcn_lib import Grapher, act_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.InstanceNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.InstanceNorm2d(out_features),
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class ViGBlock(nn.Module):
    def __init__(self, in_features, HW):
        super(ViGBlock, self).__init__()
        self.k = 9  # neighbor num (default:9)
        self.conv = 'mr'  # graph conv layer {edge, mr}
        self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = 'instance'  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = 0.0
        self.HW = HW
        # self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
        # self.channels = [80, 160, 400, 640]  # number of channels of deep features

        # self.n_blocks = sum(self.blocks)  # 12(Pyramid-ti)
        # channels = opt.channels  # [48, 96, 240, 384](Pyramid-ti)-每个 stage 的特征维度
        # reduce_ratios = [4, 2, 1, 1]
        # dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_blocks)]  # stochastic depth decay rule
        # num_knn = [int(x.item()) for x in torch.linspace(self.k, self.k, self.n_blocks)]  # number of knn's k
        # max_dilation = 49 // max(num_knn)
        vig_block = [
            Grapher(in_channels=in_features, kernel_size=9, dilation=1, conv=self.conv, act=self.act,
                    norm=self.norm,
                    bias=self.bias, stochastic=self.use_stochastic, epsilon=self.epsilon, r=1, n=self.HW,
                    drop_path=self.drop_path,
                    relative_pos=True),
            FFN(in_features, in_features * 4, act=self.act, drop_path=self.drop_path),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            Grapher(in_features, 9, 1, self.conv, self.act, self.norm,
                    self.bias, self.use_stochastic, self.epsilon, r=1, n=self.HW, drop_path=self.drop_path,
                    relative_pos=True),
            FFN(in_features, in_features * 4, act=self.act, drop_path=self.drop_path),
            nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*vig_block)

    def forward(self, x):
        # y = self.conv_block(x)
        return x + self.conv_block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# distribution generator
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks=1, n_vig_blocks=1):
        super(Generator, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, 256 // 4, 256 // 4))
        self.HW = 256 // 4 * 256 // 4

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        vig_blocks = []
        for _ in range(n_vig_blocks):
            vig_blocks += [ViGBlock(in_features, self.HW)]
        model_head += vig_blocks
        # Residual blocks
        model_body = []
        for _ in range(n_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x) + self.pos_embed
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


def define_Generator(input_nc, output_nc, norm='instance', init_type='normal', init_gain=0.02):
    net = Generator(input_nc, output_nc, 9, n_vig_blocks=2)
    init_weights(net, 'kaiming', init_gain)
    return net


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),  # out_dim / 2 * img_size / 2 * img_size / 2
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),  # out_dim * img_siz / 4 * img_size / 4
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),  # out_dim * img_size / 4 * img_size / 4
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
