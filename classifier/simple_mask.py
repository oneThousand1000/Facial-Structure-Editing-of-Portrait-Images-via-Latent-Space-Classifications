
from feature_extractor.neck_mask_extractor import  get_simple_neck_mask
import os
import cv2
train=open('./data/train/mask.flist','w')
val=open('./data/val/mask.flist','w')


with open('./data/train/origin_img.flist','r') as f:
    for line in f.readlines():
        path = line[:-3]
        name = os.path.basename(path)
        save_path = os.path.join('F:/DoubleChin/datasets/CelebAMask-HQ/neck_mask', name.replace('jpg', 'png'))
        # if not os.path.exists(save_path):
        #
        #     mask=get_simple_neck_mask(path)
        #
        #
        #
        #     cv2.imwrite(save_path,mask)
        train.write(save_path+'\n')

with open('./data/val/origin_img.flist','r') as f:
    for line in f.readlines():
        # path=line[:-3]
        # mask=get_simple_neck_mask(path)
        name=os.path.basename(path)

        save_path=os.path.join('F:/DoubleChin/datasets/CelebAMask-HQ/neck_mask',name.replace('jpg','png'))
        # cv2.imwrite(save_path,mask)
        val.write(save_path+'\n')