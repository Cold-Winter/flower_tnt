import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import scipy.io as scio



class FlowerDataSet(data.Dataset):
    def __init__(self, root_path='../flower_jpg/',indexlist = None, labellist = None, image_tmpl = "image_{:05d}.jpg" ,
                    transform=None,split=None,
                   test_mode=False):
        self.imagelist = indexlist
        self.directory = root_path
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.labellist = labellist
        


    def _load_image(self, directory, idx):

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]


    def __getitem__(self, index):
        imageid = self.imagelist[index]
        images = list()
        image = self._load_image(self.directory, imageid)
        images.extend(image)
        process_data = self.transform(images)
        return process_data, self.labellist[imageid]

    def __len__(self):
        return len(self.imagelist)
