import os
import glob
import random
import cv2
from src.feature_extractor.neck_mask_extractor import  get_simple_neck_mask


for img in glob.glob('F:/DoubleChin/datasets/CelebAMask-HQ/generated_double_chin_img/generate_img/*.jpg'):
    name = os.path.basename(img)[:-4]

    if not os.path.exists(f'F:/DoubleChin/datasets/CelebAMask-HQ/generated_double_chin_img/mask/{name}.png'):
        mask = get_simple_neck_mask(img=img)
        print(img)
        cv2.imwrite(f'F:/DoubleChin/datasets/CelebAMask-HQ/generated_double_chin_img/mask/{name}.png',mask)




train=open('./data/train/origin_img.flist','w')
val=open('./data/val/origin_img.flist','w')
chin=open('./double_chin_imgs.txt','r')
split=0.9

lines=[]

for img in glob.glob('F:/DoubleChin/datasets/CelebAMask-HQ/generated_double_chin_img/generate_img/*.jpg')\
        +glob.glob('F:/DoubleChin/datasets/CelebAMask-HQ/generated_double_chin_img/generate_img/*.png'):



    lines.append(img+' 1')
#
# double_chins=chin.readlines()
#
# count=0
#normal_face
for img in glob.glob('F:/DoubleChin/datasets/CelebAMask-HQ/CelebA-HQ-img/*.jpg'):
    name = os.path.basename(img)
    if name not  in double_chins:

        lines.append(img + ' 0')
        count+=1
        if(count>3000):
            break

random.shuffle(lines)

for line in lines:
    if random.randint(0,100)%19==0:
        val.write(line+'/n')
    else:
        train.write(line+'/n')