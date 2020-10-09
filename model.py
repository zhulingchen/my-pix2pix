import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, n_outer_channels, n_inner_channels, n_input_channels=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            n_outer_channels (int): the number of filters in the outer conv layer
            n_inner_channels (int): the number of filters in the inner conv layer
            n_input_channels (int): the number of channels in input images/features
            submodule (UnetSkipConnectionBlock): previously defined submodules
            outermost (bool): if this module is the outermost module
            innermost (bool): if this module is the innermost module
            norm_layer (nn.Module): normalization layer
            use_dropout (bool): if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = (norm_layer == nn.InstanceNorm2d)
        if n_input_channels is None:
            n_input_channels = n_outer_channels
        downconv = nn.Conv2d(n_input_channels, n_inner_channels,
                             kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(n_inner_channels)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(n_outer_channels)

        if outermost:
            upconv = nn.ConvTranspose2d(n_inner_channels * 2, n_outer_channels,
                                        kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(n_inner_channels, n_outer_channels,
                                        kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(n_inner_channels * 2, n_outer_channels,
                                        kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # add skip connections by concatenation on the channel axis in the non-outermost blocks
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)


class Pix2pixGenerator(nn.Module):
    """Define a Unet-based generator"""
    def __init__(self, n_input_channels, n_output_channels,
                 num_downs, n_first_conv_filters=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a U-net generator
        Parameters:
            n_input_channels (int): the number of channels in input images
            n_output_channels (int): the number of channels in output images
            num_downs (int): the number of downsamplings in UNet.
                             For example, if |num_downs| == 7, image of size 128x128 will become of size 1x1 # at the bottleneck
            n_first_conv_filters (int): the number of filters in the last conv layer
            norm_layer: normalization layer

        Construct the U-net from the innermost layer to the outermost layer
        It is a recursive process.
        """
        super(Pix2pixGenerator, self).__init__()
        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(n_first_conv_filters * 8, n_first_conv_filters * 8,
                                             innermost=True, norm_layer=norm_layer)
        # add intermediate layers with n_first_conv_filters * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(n_first_conv_filters * 8, n_first_conv_filters * 8,
                                                 submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from n_first_conv_filters * 8 to n_first_conv_filters
        unet_block = UnetSkipConnectionBlock(n_first_conv_filters * 4, n_first_conv_filters * 8,
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(n_first_conv_filters * 2, n_first_conv_filters * 4,
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(n_first_conv_filters, n_first_conv_filters * 2,
                                             submodule=unet_block, norm_layer=norm_layer)
        # add the outermost layer
        self.model = UnetSkipConnectionBlock(n_output_channels, n_first_conv_filters, n_input_channels=n_input_channels,
                                             submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input_src):
        return self.model(input_src)


class Pix2pixDiscriminator(nn.Module):
    """Define a PatchGAN discriminator"""
    def __init__(self, n_input_channels, n_first_conv_filters=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            n_input_channels (int): the number of channels in input images
            n_first_conv_filters (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer (nn.Module): normalization layer
        """
        super(Pix2pixDiscriminator, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)
        sequence = [nn.Conv2d(n_input_channels, n_first_conv_filters,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        # gradually increase the number of filters
        for n in range(1, n_layers+1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(n_first_conv_filters * nf_mult_prev, n_first_conv_filters * nf_mult,
                          kernel_size=4, stride=2 if n < n_layers else 1, padding=1, bias=use_bias),
                norm_layer(n_first_conv_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # output 1 channel prediction map
        sequence += [nn.Conv2d(n_first_conv_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input_src, input_tgt):
        x = torch.cat([input_src, input_tgt], dim=1)
        return self.model(x)


class Pix2pixGAN(nn.Module):
    pass