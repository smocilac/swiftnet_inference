import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

upsample = lambda x, size: F.interpolate(x, size, mode='nearest')
batchnorm_momentum = 0.01 / 2


# util_image_size=(128, 256)
# util_image_size=(256, 512)
# util_image_size=(512, 1024)
util_image_size=(1024, 2048)
util_target_size=(util_image_size[0] // 4, util_image_size[1] // 4)


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)
        self.num_maps_in = num_maps_in
        self.num_maps_out = num_maps_out
        self.skip_maps_in = skip_maps_in
        #self.x = 64 * (256 // skip_maps_in)
        self.x = 8 * (util_target_size[0] * 8 // skip_maps_in)
        self.y = self.x * 2

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        #skip_size = skip.size()[2:4]
        #print("skip.size()[2:4]", skip.size()[2:4])
        skip_size = (self.x, self.y)
        #print("skip size = ", skip_size, " ", self.num_maps_in, " ", self.num_maps_out, " ", self.skip_maps_in)
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        # target_size = x.size()[2:4]
        # target_size = (32,64)
        target_size=(util_target_size[0] // 8, util_target_size[1] // 8)
        #print ("x.size()[2:4] = ", x.size()[2:4])
        #print ("target_size", target_size)
        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1
        #print(num," ", target_size)
        # grid_sizes = [(8,16), (4,8), (2,4)]

        for i in range(1, num):
            if not self.square_grid:
                #pdb.set_trace()
                grid_size = (self.grids[i - 1], max(1, round(float(ar * self.grids[i - 1]))))
                # print("grid size = ", grid_sizes[i-1])
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)# grid_sizes[i-1])
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            #level = upsample(level, target_size)
            level = F.interpolate(level, target_size, mode='nearest')
            levels.append(level)
        #print("forward passed")
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=3, batch_norm=use_bn)

    def forward(self, x, skip):
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x
