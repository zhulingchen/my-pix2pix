import os
import argparse
import torch

from model import *



if __name__ == '__main__':
    # define input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        required=True, type=str, help="config file")
    parser.add_argument('-d', '--dataset',
                        required=True, type=str, help="dataset name")
    parser.add_argument('-v', '--verbose',
                        action='store_true', help="verbosity")
    args = parser.parse_args()

    # build pix2pix model architecture
    model = Pix2pixGAN(args)
    model.load_config()
    model.load_dataset()
    model.build_discriminator()
    model.build_generator()

    pass