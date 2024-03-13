
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
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
    def __init__(self, root, mode='train'):
        super().__init__()
        self.root = os.path.join(root, mode)
        # sorted function takes an iterable as an input and returns a new list containing all items from the iterable in ascending order
        self.classes = sorted(os.listdir(self.root)) # ['car', 'motorcycle', 'truck']
        # print(f'classes: {self.classes}')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths = []
        self.img_labels = []
        for cls_name in self.classes:
            cls_path = os.path.join(self.root, cls_name)
            for img_path in glob(os.path.join(cls_path, '*.jpg')):
                self.img_paths.append(img_path)
                self.img_labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.img_labels[index]
        image = imgloader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    

