import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

def extract_hog_features(img_path):
    image = imread(img_path)
    image_resized = resize(image, (64, 64))  # Resize to standardize size
    fd, hog_image = hog(image_resized, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd

def get_key(fp):
    filename = fp.split('\\')[-1]
    # print(f'Filename : {filename}')
    filename = filename.split('.')[0].replace('frame', '')
    # print(f'Filename : {filename}')
    return int(filename)

class VehicleDataset(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, img_paths, img_labels, transform=None):
        super().__init__()
        self.transform = transform
        self.img_paths = img_paths
        self.img_labels = img_labels


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.img_labels[index]
        image = imgloader(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label
    

