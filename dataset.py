import os
import cv2
from torch.utils.data import Dataset



class Pix2pixDataset(Dataset):
    """Construct a dataset for pix2pix GAN training"""
    def __init__(self, root_dir, transforms_src=None, transforms_tgt=None):
        """
        Parameters:
            root_dir       : root directory of input images
            transforms_src : image transformers for source images
            transforms_tgt : image transformers for target images
        """
        self.images_src, self.images_tgt = [], []
        self.root_dir = os.path.normpath(root_dir)
        self.transforms_src = transforms_src
        self.transforms_tgt = transforms_tgt
        images = self.__load_images()
        self.length = len(images)
        for image in images:
            _, n_cols, _ = image.shape
            self.images_src.append(image[:, n_cols//2:])
            self.images_tgt.append(image[:, :n_cols//2])

    def __load_images(self):
        images = []
        for filename in os.listdir(self.root_dir):
            image = cv2.imread(os.path.join(self.root_dir, filename))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2.imread works with the BGR order
                images.append(image)
        return images

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image_src = self.images_src[item]
        image_tgt = self.images_tgt[item]
        if self.transforms_src:
            image_src = self.transforms_src(image_src)
        if self.transforms_tgt:
            image_tgt = self.transforms_tgt(image_tgt)
        return image_src, image_tgt