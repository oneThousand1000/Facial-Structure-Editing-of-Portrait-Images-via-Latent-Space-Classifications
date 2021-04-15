import numpy as np
from src.feature_extractor.neck_mask_extractor import  get_simple_neck_mask
import os
import cv2

img1='F:/CHINGER/docs/ffhq/00016.jpg'
mask1=get_simple_neck_mask(img1)
mask1=cv2.resize(mask1,(1024,1024))
cv2.imwrite('F:/CHINGER/docs/paper_imgs/classifier_imgs/00016_masked.jpg',cv2.imread(img1)*(mask1//255))
img2='F:/CHINGER/docs/ffhq/00158.jpg'
mask2=get_simple_neck_mask(img2)
mask2=cv2.resize(mask2,(1024,1024))
cv2.imwrite('F:/CHINGER/docs/paper_imgs/classifier_imgs/00158_masked.jpg',cv2.imread(img2)*(mask2//255))
cv2.imwrite('F:/CHINGER/docs/paper_imgs/classifier_imgs/classifier-data-temp.jpg',np.concatenate([cv2.imread(img1),cv2.imread(img1)*(mask1//255),cv2.imread(img2),cv2.imread(img2)*(mask2//255)],axis=1))
