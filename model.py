import math
import torch
from torch import nn
from termcolor import colored
from torch import nn
from collections import OrderedDict
from torchvision import models
from termcolor import colored


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return block8
        #return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class TCD_ResNet(nn.Module):
    def __init__(self, args, dset_classes, fc_in = 2048):
        super(TCD_ResNet, self).__init__()
        self.args = args
        self.dset_classes = dset_classes
        self.fc_in = fc_in

    def load_resnet50_layers(self, args, layers):
        resnet50 = models.resnet50(True)
        child_modules = resnet50.named_children()
        trunk_list = OrderedDict({})
        for name, module in child_modules:
            if name in layers:
                trunk_list[name] = module
        #pdb.set_trace()
        return nn.Sequential(trunk_list)

    def adjust_cap(self, cap, n):
        cap.fc = nn.Linear(cap.fc.in_features, n)
        return cap

    def freeze_layers(self, freeze_layers):
        if freeze_layers == 1:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
        elif freeze_layers == 2:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']
        elif freeze_layers == 3:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
        elif freeze_layers == 4:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        else:
            print('Invalid parameters for args.freeze:%s in %s' % (freeze_layers, self.__class__.__name__))
            exit()
        trunk_layers = self.trunk.named_children()
        # pdb.set_trace()

        for layer in trunk_layers:
            if layer[0] in freeze_layer_list:
                print(colored('Freezing Layer : %s' % (layer[0]), 'white'))
                layer_parameters = layer[1].parameters()
                for param in layer_parameters:
                    param.requires_grad = False

    def create_cap(self):
        fc_in = self.fc_in
        if self.args.FC3:
            cap_A2 = nn.Sequential(OrderedDict([
                ('layer_1', nn.Sequential(nn.Linear(fc_in, fc_in), \
                                          nn.BatchNorm1d(fc_in), nn.PReLU(), nn.Dropout(p=0.5))), \
                ('layer_2', nn.Sequential(nn.Linear(fc_in, int(fc_in / 2)), \
                                          nn.BatchNorm1d(int(fc_in / 2)), nn.PReLU(),
                                          nn.Dropout(p=0.5))), \
                ('layer_3', nn.Sequential(nn.Linear(int(fc_in / 2), int(fc_in / 8)), \
                                          nn.PReLU(), nn.Dropout(p=0.75))), \
                ('layer_final', nn.Sequential(nn.Linear(int(fc_in / 8), len(self.dset_classes))))
            ]))
        else:
            cap_A2 = nn.Sequential(OrderedDict([
                ('layer_final', nn.Sequential(nn.Linear(fc_in, len(self.dset_classes))))
            ]))

        return cap_A2

class Model_A1(TCD_ResNet):
    def __init__(self, args = None, dset_classes = 1000):
        super(Model_A1, self).__init__(args, dset_classes)
        #pdb.set_trace()
        #self.trunk = self.load_resnet50_layers(args, ['conv1', 'bn1', 'relu', \
        #'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'])
        self.base = self.load_resnet50_layers(args, ['conv1', 'bn1', 'relu'])
        self.trunk =self.load_trunk(args)
        self.decoder = Model_A1_Decoder_3(args)
        #if freeze_layers != None:
        self.freeze_layers(1)


    def forward(self, x):
        x = x.cuda()
        #pdb.set_trace()
        # if self.training:
        o_b = self.base(x)
        o_t = self.trunk(o_b)
        y_h, y_A1 = self.decoder(o_t)
        #return {'A1': y_A1, 'input': x.data }
        #return {'A1': y_A1}
        return y_A1

    def load_trunk(self, args = None):
        layer_list = ['layer1', 'layer2', 'layer3', 'layer4']
        load_layer_list = [layer_list[i] for i in range(1)]
        return self.load_resnet50_layers(args, load_layer_list)

    def freeze_layers(self, freeze_layers):
        if freeze_layers == 1:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
        elif freeze_layers == 2:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']
        elif freeze_layers == 3:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
        elif freeze_layers == 4:
            freeze_layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        else:
            print('Invalid parameters for args.freeze:%s in %s' % (freeze_layers, self.__class__.__name__))
            exit()
        trunk_layers = self.trunk.named_children()
        base_layers = self.base.named_children()
        # pdb.set_trace()

        for layer in trunk_layers:
            if layer[0] in freeze_layer_list:
                print(colored('Freezing Layer : %s' % (layer[0]), 'white'))
                layer_parameters = layer[1].parameters()
                for param in layer_parameters:
                    param.requires_grad = False

        for layer in base_layers:
            if layer[0] in freeze_layer_list:
                print(colored('Freezing Layer : %s' % (layer[0]), 'white'))
                layer_parameters = layer[1].parameters()
                for param in layer_parameters:
                    param.requires_grad = False

class Model_A1_Decoder_3(nn.Module):
    #Superesolution decoder
    def __init__(self, args = None):
        super(Model_A1_Decoder_3, self).__init__()

        self.residual_layers = nn.Sequential(*[ResidualBlock(256) for _ in range(5)])
        self.intermediate_bn_layer = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=3, padding=1),\
                                              nn.BatchNorm2d(256)])
        self.upsample_layer = nn.Sequential(*[NC_UpsampleBLock_1(256, up_scale = 2 * 2)])
        #self.feature_to_image_layer = nn.Sequential(*[nn.Conv2d(256,\
        #                                                       3, kernel_size=9, padding=4), nn.Tanh()])
        #self.feature_to_image_layer = nn.Sequential(*[nn.Conv2d(256,\
         #                                                      3, kernel_size=9, padding=4)])

        #pdb.set_trace()

    def forward(self, x):
        #pdb.set_trace()
        o_r = self.residual_layers(x)
        o_bn = self.intermediate_bn_layer(o_r)
        y = self.upsample_layer(x + o_bn)
        #y = torch.clamp(self.feature_to_image_layer(o_us), -1, 1)
        #y = self.feature_to_image_layer(o_us)
        return o_bn, y

class NC_UpsampleBLock_1(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(NC_UpsampleBLock_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, 48, kernel_size=1, padding=0)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)

    def forward(self, x):
        #pdb.set_trace()
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

