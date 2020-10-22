import argparse

from model import *



if __name__ == '__main__':
    # define input arguments
    # train example: train -c config.yaml -d facades
    # test example: test -c config.yaml -d facades -i datasets/facades/test/1_src.jpg datasets/facades/test/2_src.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str.lower, choices=['train', 'test'], help='work mode')
    parser.add_argument('-c', '--config', required=True, type=str, help="config file")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="dataset name")
    parser.add_argument('-i', '--input', type=str, nargs='*', help="input source image (test mode only)")
    args = parser.parse_args()
    if (args.mode == 'test') and ('input' not in args):
        parser.error('At least one input image must be provided for the test mode.')

    # build pix2pix model architecture
    model = Pix2pixGAN(args)
    if model.is_train:
        model.train()
        model.save_models()
    else:
        model.load_models()
        model.test()