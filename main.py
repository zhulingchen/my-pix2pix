import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

from dataset import *
from model import *



if __name__ == '__main__':
    # load dataset
    dataset = 'facades'
    train_path = 'datasets/{:s}/train'.format(dataset)
    transforms_src = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transforms_tgt = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = Pix2pixDataset(train_path, transforms_src, transforms_tgt)
    print('Loaded {:d} training samples from {:s}'.format(len(train_dataset), train_path))

    # build discriminator
    assert all(s[0].shape == s[1].shape for s in train_dataset) and (len(set(s[0].shape for s in train_dataset)) == 1), "The shape of all source and target images must be the same."
    discriminator = Pix2pixDiscriminator(train_dataset[0][0].shape[0] * 2)
    print('Pix2pix discriminator architecture')
    summary(discriminator, [s.numpy().shape for s in train_dataset[0]], device='cpu')

    # build generator
    generator = Pix2pixGenerator(train_dataset[0][0].shape[0], train_dataset[0][1].shape[0], num_downs=8, use_dropout=True)
    print('Pix2pix generator architecture')
    summary(generator, train_dataset[0][0].numpy().shape, device='cpu')

    pass