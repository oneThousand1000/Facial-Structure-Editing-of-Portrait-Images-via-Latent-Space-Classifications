#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""
import torch
import torchvision.transforms as transforms
from . import mobilenet_v1
import numpy as np
import cv2
import dlib
from .utils.ddfa import ToTensorGjz, NormalizeGjz
from .utils.inference import crop_img, predict_68pts,parse_roi_box_from_bbox
import torch.backends.cudnn as cudnn
import os.path as osp
STD_SIZE = 120

def get_3DDFA_model():
    # 1. load pre-tained model

    checkpoint_fp =  osp.join(osp.dirname(osp.realpath(__file__)),  'models/phase1_wpdc_vdc.pth.tar')#'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    cudnn.benchmark = True
    model = model.cuda()
    model.eval()
    return model


def threeDDFA_landmarks_extractor(image,model):
    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])



    if isinstance(image,str):
        img_ori = cv2.imread(image)
    else:
        img_ori = image


    rects = face_detector(img_ori, 1)

    assert len(rects) != 0, 'no face detected!'
    rect=rects[0]
    bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
    roi_box = parse_roi_box_from_bbox(bbox)
    img = crop_img(img_ori, roi_box)

    # forward: one step
    img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    with torch.no_grad():
        input = input.cuda()
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    # 68 pts
    pts68 = predict_68pts(param, roi_box)
    pts68 = np.round(pts68).astype(np.int32)
    pts68=pts68[:2,:]
    pts68 = pts68.transpose(1,0)
    return pts68







