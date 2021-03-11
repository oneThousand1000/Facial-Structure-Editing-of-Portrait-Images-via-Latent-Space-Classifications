import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .feature_extractor.neck_mask_extractor import  get_simple_neck_mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self,config, img_flist,mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data_dir=config.data_dir
        #print('img_flist: ',img_flist)
        self.img_data = self.load_flist(img_flist)
        self.mask_data=self.load_flist(mask_flist)
        #print(self.img_data)


        self.input_size = config.INPUT_SIZE



    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        item = self.load_item(index)

        return item

    def load_name(self, index):
        name = self.img_data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        image_path=self.img_data[index][:-3]
        label=int(self.img_data[index][-2])

        if(label==1):
            label=[0,1]
        else:
            label=[1,0]

        img = imread(image_path)

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)


        mask = self.load_mask(index,size)


        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]


        return self.to_tensor(img),self.to_tensor(mask),torch.tensor(label).float()

    def generate_mask(self,img,size):
        mask=get_simple_neck_mask(img=img)
        mask = self.resize(mask, size, size, centerCrop=False)
        mask = rgb2gray(mask)
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def load_mask(self,index,size):
        mask = imread(self.mask_data[index][:-1])
        mask = self.resize(mask, size, size, centerCrop=False)
        mask = rgb2gray(mask)
        mask = (mask > 0).astype(np.uint8) * 255
        return mask


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                return open(flist,'r').readlines()

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
    def generate_test_data(self,img):
        if isinstance(img,str):
            img = imread(img)
        else:
            img= img
        size = self.input_size
        img = self.resize(img, size, size)
        mask=self.generate_mask(img,size)
        return self.to_tensor(img).unsqueeze(0).cuda(), self.to_tensor(mask).unsqueeze(0).cuda()
