import argparse

from model import *



if __name__ == '__main__':
    # define input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str.lower, choices=['train', 'test'], help='an integer for the accumulator')
    parser.add_argument('-c', '--config', required=True, type=str, help="config file")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="dataset name")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbosity")
    args = parser.parse_args()

    # build pix2pix model architecture
    model = Pix2pixGAN(args)
    if model.is_train:
        model.train()