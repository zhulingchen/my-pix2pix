# pix2pix: Image-to-Image Translation with Conditional Adversarial Nets

## Requirements

* numpy
* PyTorch >= 1.4.0
* opencv-python
* [torchsummary](https://github.com/sksq96/pytorch-summary)

## Usage

main.py [-h] -c CONFIG -d DATASET [-g GENERATE] [-s SAVEFILE] {train,test}

| Arguments     	                                  | Description                         |
| :---          				                      | :---                                |
| -h, --help                                          | show this help message and exit     |
| -c CONFIG, --config CONFIG                          | config file                 	    |
| -d DATASET, --dataset DATASET                       | dataset name                        |
| -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]] | input source image (test mode only) |
| {train,test}                                        | work mode                           |

## Reference

* [Website](https://phillipi.github.io/pix2pix/)
* [Original paper](https://arxiv.org/abs/1611.07004)
* [Authors' implementation (PyTorch 1.4)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)