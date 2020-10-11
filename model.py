import os
import yaml
import time
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import init
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
from PIL import Image

from dataset import *



def get_norm_layer(name):
    name = name.lower()
    if name == 'batch':
        return nn.BatchNorm2d
    elif name == 'instance':
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError('Normalization layer name {:s} is not supported.'.format(name))


def get_gan_loss(name, device):
    name = name.lower()
    if name == 'vanilla':
        def bce_with_logits_and_singleton_target_loss(input, target):
            assert isinstance(input, torch.Tensor) and isinstance(target, (bool, int, float))
            target_tensor = torch.tensor(target).expand_as(input).float().to(device)
            return F.binary_cross_entropy_with_logits(input, target_tensor)
        return bce_with_logits_and_singleton_target_loss
    elif name == 'wgangp':
        def wgangp_loss(input, target):
            assert isinstance(input, torch.Tensor) and isinstance(target, (bool, int, float))
            return -input.mean() if bool(target) else input.mean()
        return wgangp_loss
    else:
        raise NotImplementedError('GAN loss name {:s} is not supported.'.format(name))


def denormalize_image(image):
    assert isinstance(image, torch.Tensor)
    image_numpy = (np.transpose(image.cpu().numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        + -------------------identity--------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, n_outer_channels, n_inner_channels, n_input_channels=None,
                 submodule=None, outermost=False, innermost=False, norm_layer='batch_norm', use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            n_outer_channels (int): the number of filters in the outer conv layer
            n_inner_channels (int): the number of filters in the inner conv layer
            n_input_channels (int): the number of channels in input images/features
            submodule (UnetSkipConnectionBlock): previously defined submodules
            outermost (bool): if this module is the outermost module
            innermost (bool): if this module is the innermost module
            norm_layer (str): normalization layer name
            use_dropout (bool): if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        norm_layer = get_norm_layer(norm_layer)
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
                                        kernel_size=4, stride=2, padding=1)  # in_channels is doubled because of the previous concatenation
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
                                        kernel_size=4, stride=2, padding=1, bias=use_bias)  # in_channels is doubled because of the previous concatenation
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
    def __init__(self, n_input_channels, n_output_channels, num_downs,
                 n_first_conv_filters=64, norm_layer='batch_norm', use_dropout=False):
        """Construct a U-net generator
        Parameters:
            n_input_channels (int): the number of channels in input images
            n_output_channels (int): the number of channels in output images
            num_downs (int): the number of downsamplings in UNet.
                             For example, if |num_downs| == 7, image of size 128x128 will become of size 1x1 # at the bottleneck
            n_first_conv_filters (int): the number of filters in the last conv layer
            norm_layer (str): normalization layer name

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
    def __init__(self, n_input_channels, n_first_conv_filters=64, n_layers=3, norm_layer='batch_norm'):
        """Construct a PatchGAN discriminator

        Parameters:
            n_input_channels (int): the number of channels in input images
            n_first_conv_filters (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer (str): normalization layer name
        """
        super(Pix2pixDiscriminator, self).__init__()
        norm_layer = get_norm_layer(norm_layer)
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


class Pix2pixGAN():
    """Define a Pix2pix GAN"""
    def __init__(self, args):
        """Construct a Pix2pix GAN

        Parameters:
            args (argparse.Namespace): argument list
        """
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.config = args.config
        self.dataset = args.dataset
        self.verbose = args.verbose
        self.is_train = (args.mode == 'train')
        self.__load_config()
        self.__build_generator()
        if self.is_train:
            self.__load_dataset()
            self.__build_discriminator()
            self.gan_loss = get_gan_loss(self.config['loss'], self.device)
            self.l1_loss = nn.L1Loss()
            self.opt_g = torch.optim.Adam(self.generator.parameters(),
                                          lr=self.config['lr'],
                                          betas=(self.config['beta1'], self.config['beta2']))
            self.opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                          lr=self.config['lr'],
                                          betas=(self.config['beta1'], self.config['beta2']))

    def __init_weights(self, net, type='normal', gain=0.02):
        """Initialize network weights

        Parameters:
            net (network)   -- network to be initialized
            type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)    -- scaling factor for normal, xavier and orthogonal.

        Initialization type 'normal' was used in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        # apply the initialization function <init_func>
        net.apply(init_func)

    def __load_config(self):
        with open(self.config, 'r') as f:
            self.config = yaml.safe_load(f)

    def __load_image_transforms(self):
        self.transforms_src = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((self.config['image_rows'], self.config['image_cols'])),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.transforms_tgt = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((self.config['image_rows'], self.config['image_cols'])),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __load_dataset(self):
        train_dataset_path = 'datasets/{:s}/train'.format(self.dataset)
        val_dataset_path = 'datasets/{:s}/val'.format(self.dataset)
        if not os.path.exists(train_dataset_path):
            raise ValueError('Train image directory {:s} does not exist.'.format(train_dataset_path))
        if not os.path.exists(val_dataset_path):
            self.use_val = False
            warnings.warn('Validation image directory {:s} does not exist.'.format(val_dataset_path))
        self.__load_image_transforms()
        train_dataset = Pix2pixDataset(train_dataset_path, self.transforms_src, self.transforms_tgt)
        assert all(s[0].shape == s[1].shape for s in train_dataset) and (len(set(s[0].shape for s in train_dataset)) == 1), \
            "The shape of all source and target images must be the same."
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        print('Loaded {:d} training samples from {:s} '\
              '(batch size: {:d}, number of batches: {:d})'.format(len(train_dataset), train_dataset_path,
                                                                   self.config['batch_size'], len(self.train_dataloader)))
        if os.path.exists(val_dataset_path):
            self.use_val = True
            val_dataset = Pix2pixDataset(val_dataset_path, self.transforms_src, self.transforms_tgt)
            assert all(s[0].shape == s[1].shape for s in val_dataset) and (len(set(s[0].shape for s in val_dataset)) == 1), \
                "The shape of all source and target images must be the same."
            self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
            print('Loaded {:d} validation samples from {:s} ' \
                  '(batch size: {:d}, number of batches: {:d})'.format(len(val_dataset), val_dataset_path,
                                                                       1, len(self.val_dataloader)))

    def __build_generator(self):
        self.generator = Pix2pixGenerator(n_input_channels=self.config['image_chns'],
                                          n_output_channels=self.config['image_chns'],
                                          num_downs=self.config['generator_downsamplings'],
                                          n_first_conv_filters=self.config['generator_first_conv_filters'],
                                          norm_layer=self.config['norm_layer'],
                                          use_dropout=self.config['use_dropout'])
        # initialize network weights
        print('Initialize generator network with {:s}'.format(self.config['init_type']))
        self.__init_weights(self.generator, self.config['init_type'], self.config['init_gain'])
        if self.verbose:
            print('Pix2pix generator architecture')
            summary(self.generator, (self.config['image_chns'], self.config['image_rows'], self.config['image_cols']), device='cpu')
        self.generator = self.generator.to(self.device)

    def __build_discriminator(self):
        self.discriminator = Pix2pixDiscriminator(n_input_channels=2 * self.config['image_chns'],
                                                  n_first_conv_filters=self.config['discriminator_first_conv_filters'],
                                                  n_layers=self.config['discriminator_conv_layers'],
                                                  norm_layer=self.config['norm_layer'])
        # initialize network weights
        print('Initialize discriminator network with {:s}'.format(self.config['init_type']))
        self.__init_weights(self.discriminator, self.config['init_type'], self.config['init_gain'])
        if self.verbose:
            print('Pix2pix discriminator architecture')
            summary(self.discriminator, [(self.config['image_chns'], self.config['image_rows'], self.config['image_cols'])] * 2, device='cpu')
        self.discriminator = self.discriminator.to(self.device)

    def train(self):
        train_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            for batch, (real_src, real_tgt, _) in enumerate(self.train_dataloader):
                real_src = real_src.to(self.device)
                real_tgt = real_tgt.to(self.device)
                # generate fake target
                fake_tgt = self.generator(real_src)
                # update discriminator
                for param in self.discriminator.parameters():  # enable backprop for discriminator
                    param.requires_grad = True
                self.opt_d.zero_grad()  # clear discriminator gradients
                pred_fake = self.discriminator(real_src, fake_tgt.detach())  # discriminate fake; stop backprop to the generator
                loss_d_fake = self.gan_loss(pred_fake, False)  # discriminator loss on fake
                pred_real = self.discriminator(real_src, real_tgt)  # discriminate real
                loss_d_real = self.gan_loss(pred_real, True)  # discriminator loss on real
                loss_d = 0.5 * (loss_d_fake + loss_d_real)
                loss_d.backward()
                self.opt_d.step()  # update discriminator weights
                # update generator
                for param in self.discriminator.parameters():  # disable backprop for discriminator
                    param.requires_grad = False
                self.opt_g.zero_grad()  # clear generator gradients
                pred_fake = self.discriminator(real_src, fake_tgt)  # discriminate fake
                loss_g_gan = self.gan_loss(pred_fake, True)  # gan loss on fake; let discriminator think fake_tgt is real
                loss_g_l1 = self.config['lambda_l1'] * F.l1_loss(fake_tgt, real_tgt)  # weighted L1-loss
                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                self.opt_g.step()  # update generator weights
            print('Epoch {:d} / {:d} \t Elapsed Time: {:.4f} sec'.format(epoch+1, self.config['epochs'], time.time() - epoch_start_time))
            if ((epoch + 1) % self.config['val_freq'] == 0) and self.use_val:
                val_output_path = 'datasets/{:s}/val_output/{:s}'.format(self.dataset, train_start_time)
                if not os.path.exists(val_output_path):
                    os.makedirs(val_output_path)
                real_src, real_tgt, real_path = next(iter(self.val_dataloader))  # batch dimension shape is 1
                with torch.no_grad():
                    fake_tgt = self.generator(real_src.to(self.device))
                real_src = denormalize_image(real_src[0])
                fake_tgt = denormalize_image(fake_tgt[0])
                real_tgt = denormalize_image(real_tgt[0])
                real_path = real_path[0]
                real_filename = real_path.split(os.sep)[-1]
                real_filename_base, real_filename_ext = os.path.splitext(real_filename)
                real_filename_base = 'epoch_{:d}_{:s}'.format(epoch + 1, real_filename_base)
                val_output_image = np.concatenate([real_src, fake_tgt, real_tgt], axis=1)
                val_output_image = Image.fromarray(val_output_image, 'RGB')
                val_output_image.save(os.path.join(os.path.normpath(val_output_path), real_filename_base + real_filename_ext))

    def save_networks(self):
        pass