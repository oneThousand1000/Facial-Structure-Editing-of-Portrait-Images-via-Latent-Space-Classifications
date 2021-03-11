#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .face_parsing_PyTorch.model import BiSeNet
import sys
sys.path.append('./')
import torch
import skimage.morphology as morphology
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from .chin_edge_extractor import chin_mask_extractor,warpping_area,chin_edge_extractor

def paring_face_mask(parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255




    pi = 14

    index = np.where(vis_parsing_anno == pi)
    neck_mask = np.zeros_like(vis_parsing_anno_color)
    neck_mask[index[0], index[1], :] = part_colors

    neck_mask = neck_mask.astype(np.uint8)






    face_mask= np.zeros_like(vis_parsing_anno_color,np.uint8)
    for pi in [1,2,3,4,5,6,10,11,12,13]:
        index = np.where(vis_parsing_anno == pi)
        temp = np.zeros_like(vis_parsing_anno_color)
        temp[index[0], index[1], :] = part_colors

        temp = temp.astype(np.uint8)
        face_mask = np.bitwise_or(temp, face_mask)

    # cv2.imshow('i', face_mask)
    # cv2.waitKey(0)



    return face_mask,neck_mask


def vis_parsing_maps(parsing_anno, stride,cloth_mask=True):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    pi=14

    index = np.where(vis_parsing_anno == pi)
    neck_mask = np.zeros_like(vis_parsing_anno_color)
    neck_mask[index[0], index[1], :] = part_colors

    neck_mask = neck_mask.astype(np.uint8)
    # cv2.imshow(str(pi), neck_mask)
    # cv2.waitKey(0)
    pi = 1

    index = np.where(vis_parsing_anno == pi)
    face_mask = np.zeros_like(vis_parsing_anno_color)
    face_mask[index[0], index[1], :] = part_colors

    face_mask = face_mask.astype(np.uint8)

    # num_of_class = np.max(vis_parsing_anno)
    # full_mask= np.zeros_like(vis_parsing_anno_color,np.uint8)
    # for pi in range(1, num_of_class + 1):
    #     index = np.where(vis_parsing_anno == pi)
    #     temp = np.zeros_like(vis_parsing_anno_color)
    #     temp[index[0], index[1], :] = part_colors
    #
    #     temp = temp.astype(np.uint8)
    #     full_mask = np.bitwise_or(temp, full_mask)
    #     cv2.imshow(str(pi),temp)
    #     cv2.waitKey(0)

    other_mask = np.zeros_like(vis_parsing_anno_color,np.uint8)

    return neck_mask,other_mask,face_mask

def fullmask(parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    full_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    hair_mask=None
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        temp=np.zeros_like(vis_parsing_anno_color)
        temp[index[0], index[1], :] = part_colors
        temp = temp.astype(np.uint8)
        full_mask = np.bitwise_or(temp, full_mask)

        if pi==17:
            hair_mask=temp

    return full_mask, cv2.bitwise_and(cv2.bitwise_xor(full_mask, hair_mask), full_mask)

def get_parsingNet(cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    cur_dir = os.path.dirname(__file__)
    save_pth = os.path.join(cur_dir, os.path.join('./face_parsing_PyTorch/res/cp', cp))
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net

def get_attributes(img,img_path,net=None, cp='79999_iter.pth'):
    ori_image=img
    if net==None:

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        #print(save_pth)
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])



    with torch.no_grad():
        image =Image.open(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input= to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask,other_mask,face_mask=vis_parsing_maps(parsing, stride=1)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.resize(img, (512, 512))
    try:
        di_neck = cv2.dilate(neck_mask, kernel=kernel)
        di_skin = cv2.dilate(face_mask, kernel=kernel)
    except:
        assert Exception("can not dilate")
    overlap = np.bitwise_and(di_neck, di_skin)
    neck_canny_edge_ = cv2.Canny(neck_mask, 0, 0)


    neck_canny_edge=np.stack((neck_canny_edge_,neck_canny_edge_,neck_canny_edge_),axis=2)
    chin_edge=chin_edge_extractor(ori_image)


    index = np.where(neck_mask != 0)
    right = max(index[0])
    left = min(index[0])
    up = max(index[1])
    down = min(index[1])

    neck_edge=np.bitwise_and(np.bitwise_not(overlap),neck_canny_edge)

    index=np.where(neck_edge != 0)
    for i in range(index[0].shape[0]):
        neck_edge[index[0][i]][max(0,index[1][i]-1)]=neck_edge[index[0][i]][min(neck_mask.shape[1],index[1][i]+1)]=255



    new_neck_mask = np.zeros_like(neck_mask)
    h = int((right - left) * 1.3)
    w = int((up - down) * 1.3)
    center1 = int((right + left) / 2)
    center2 = int((up + down) / 2)
    new_neck_mask[max(0, center1 - h // 2):min(512, center1 + h // 2),
    max(0, center2 - w // 2):min(512, center2 + w // 2), :] = 255

    chin_edge=cv2.resize(chin_edge,(512,512))
    new_neck_mask = cv2.bitwise_and(cv2.bitwise_xor(new_neck_mask, other_mask), new_neck_mask)

    #new_neck_mask = cv2.dilate(new_neck_mask, kernel=kernel)


    chin_mask = chin_mask_extractor(img)
    new_neck_mask = cv2.bitwise_and(cv2.bitwise_xor(new_neck_mask, chin_mask), new_neck_mask)
    # cv2.imshow('i', img * (1 - new_neck_mask // 255)+neck_edge)
    # cv2.waitKey(0)

    return neck_mask,new_neck_mask,chin_edge,neck_edge



def get_origin_neck_edge(img_list,net=None, cp='79999_iter.pth',save_path=''):

    if net==None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('./face_parsing_PyTorch/res/cp', cp))
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for img_path in img_list:

        img=cv2.imread(img_path)
        ori_image=img
        with torch.no_grad():
            img = Image.fromarray(img)
            image = img.resize((512, 512), Image.BILINEAR)
            img_input= to_tensor(image)
            img_input = torch.unsqueeze(img_input, 0)
            img_input = img_input.cuda()
            out = net(img_input)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            neck_mask,face_mask=vis_parsing_maps(parsing, stride=1)

        kernel = np.ones((5, 5), np.uint8)

        try:
            di_neck = cv2.dilate(neck_mask, kernel=kernel)
            di_skin = cv2.dilate(face_mask, kernel=kernel)
        except:
            assert Exception("can not dilate")
        overlap = np.bitwise_and(di_neck, di_skin)
        neck_canny_edge = cv2.Canny(neck_mask, 0, 0)
        neck_edge = np.zeros_like(neck_canny_edge)


        left = 512
        right = -1
        down = 512
        up = -1
        for i in range(neck_mask.shape[0]):

            for j in range(neck_mask.shape[1]):
                if (neck_canny_edge[i][j] != 0 and overlap[i][j][0] == 0):
                    neck_edge[i][j] = 255
                    if (j - 1 >= 0):
                        neck_edge[i][j - 1] = 255
                    if (j + 1 < neck_mask.shape[1]):
                        neck_edge[i][j + 1] = 255
                if (neck_mask[i, j, 0] != 0):
                    left = min(left, i)
                    right = max(right, i)
                    down = min(down, j)
                    up = max(up, j)
        cv2.imwrite(os.path.join(save_path,os.path.basename(img_path).replace('jpg','png')),neck_edge)

def get_simple_neck_mask(img,net=None, cp='79999_iter.pth'):
    if net==None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if isinstance(img,str):
            image = Image.open(img)
        else:
            image = Image.fromarray(img)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input= to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask,other_mask,face_mask=vis_parsing_maps(parsing, stride=1,cloth_mask=False)
    new_neck_mask = np.zeros_like(neck_mask)
    try:
        index= np.where(neck_mask!=0)
        right=max(index[0])
        left=min(index[0])
        h = int((right - left) * 1.3)
        center1 = int((right + left) / 2)
        new_neck_mask[max(0, center1 - h // 2):512,:, :] = 255
    except:
        pass
    return new_neck_mask

def get_skin_mask(img_path,net=None, cp='79999_iter.pth'):
    ori_image = img = cv2.imread(img_path)
    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        # print(save_pth)
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        image = Image.open(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask, other_mask, face_mask = vis_parsing_maps(parsing, stride=1, cloth_mask=False)

    face_mask += neck_mask

    return neck_mask
def get_face_mask(img_path,net=None, cp='79999_iter.pth'):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_shape = img.shape

    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        # print(save_pth)
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if isinstance(img_path, str):
            image = Image.open(img_path)
        else:
            image = Image.fromarray(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        face_mask ,neck_mask= paring_face_mask(parsing, stride=1)
    face_mask= cv2.resize(face_mask,(img_shape[0],img_shape[1]))
    neck_mask = cv2.resize(neck_mask,(img_shape[0],img_shape[1]))
    return face_mask,neck_mask

def get_neck_mask(img_path,net=None,dilate=7, cp='79999_iter.pth'):
    if isinstance(img_path, str):
        img= cv2.imread(img_path)
    else:
        img=img_path
    img_shape=img.shape

    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        # print(save_pth)
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if isinstance(img_path,str):
            image = Image.open(img_path)
        else:
            image = Image.fromarray(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask,other_mask,face_mask=vis_parsing_maps(parsing, stride=1,cloth_mask=False)

    img = cv2.resize(img, (512, 512))

    chin_mask, threshold_chin = chin_mask_extractor(img)

    threshold = int(threshold_chin / chin_mask.shape[0] * face_mask.shape[0])
    face_mask[:threshold, :, :] = 0

    neck_mask += face_mask

    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, chin_mask), neck_mask)
    if dilate>0:
        neck_mask = cv2.dilate(neck_mask, kernel=np.ones((dilate, dilate), np.uint8))

    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, chin_mask), neck_mask)
    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, other_mask), neck_mask)

    # cv2.imshow('i', (img * (1 - neck_mask // 255) + img * (neck_mask // 255) * 0.2).astype(np.uint8))
    # cv2.waitKey(0)

    neck_mask = cv2.resize(neck_mask, (img_shape[0], img_shape[1]))


    if dilate==0:
        return neck_mask,chin_mask


    return neck_mask

def get_neck_blur_mask(img_path,net=None,dilate=7, cp='79999_iter.pth'):
    if isinstance(img_path, str):
        img= cv2.imread(img_path)
    else:
        img=img_path
    img_shape=img.shape

    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if isinstance(img_path,str):
            image = Image.open(img_path)
        else:
            image = Image.fromarray(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask,other_mask,face_mask=vis_parsing_maps(parsing, stride=1,cloth_mask=False)

    img = cv2.resize(img, (512, 512))

    chin_mask, threshold_chin = chin_mask_extractor(img)

    threshold = int(threshold_chin / chin_mask.shape[0] * face_mask.shape[0])
    face_mask[:threshold, :, :] = 0

    neck_mask += face_mask

    if dilate > 0:
        neck_mask = cv2.dilate(neck_mask, kernel=np.ones((dilate, dilate), np.uint8))
    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, chin_mask), neck_mask)

    neck_mask_removed = morphology.remove_small_objects(neck_mask.astype(bool),min_size=800,connectivity=1)

    neck_mask_removed=neck_mask_removed.astype(np.uint8)*255
    if np.max(neck_mask_removed)==255:
        neck_mask=neck_mask_removed
    else:
        print('use origin')




    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, chin_mask), neck_mask)
    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, other_mask), neck_mask)


    neck_mask = cv2.resize(neck_mask, (img_shape[0], img_shape[1]))

    neck_mask = (neck_mask > 0).astype(np.uint8) * 255
    mask_dilate = cv2.dilate(neck_mask, kernel=np.ones((15, 15), np.uint8))
    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(25, 25))
    mask_dilate_blur = neck_mask + (255 - neck_mask) // 255 * mask_dilate_blur




    return mask_dilate_blur

def get_warpping_mask(img):
    if isinstance(img, str):
        img= cv2.imread(img)
    else:
        img=img
    img_shape=img.shape


    img = cv2.resize(img, (512, 512))
    warpping_mask,chin_point = warpping_area(img)
    chin_point = int(chin_point/ warpping_mask.shape[0]*img_shape[0])
    warpping_mask = cv2.resize(warpping_mask, (img_shape[0], img_shape[1]))
    return warpping_mask,chin_point
def get_neck_mask_debug(img_path,net=None,dilate=7, cp='79999_iter.pth'):
    if isinstance(img_path, str):
        img= cv2.imread(img_path)
    else:
        img=img_path
    img_shape=img.shape

    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        # print(save_pth)
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if isinstance(img_path,str):
            image = Image.open(img_path)
        else:
            image = Image.fromarray(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask,other_mask,face_mask=vis_parsing_maps(parsing, stride=1,cloth_mask=False)

    img = cv2.resize(img, (512, 512))
    original_face_mask = face_mask.copy()
    original_neck_mask=neck_mask.copy()
    full_mask = face_mask+neck_mask
    chin_mask, threshold_chin = chin_mask_extractor(img)
    chin_edge=chin_edge_extractor(img)

    threshold = int(threshold_chin / chin_mask.shape[0] * face_mask.shape[0])
    face_mask[:threshold, :, :] = 0

    neck_mask += face_mask


    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, chin_mask), neck_mask)
    if dilate>0:
        neck_mask = cv2.dilate(neck_mask, kernel=np.ones((dilate, dilate), np.uint8))

    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, chin_mask), neck_mask)
    neck_mask = cv2.bitwise_and(cv2.bitwise_xor(neck_mask, other_mask), neck_mask)

    # cv2.imshow('i', (img * (1 - neck_mask // 255) + img * (neck_mask // 255) * 0.2).astype(np.uint8))
    # cv2.waitKey(0)

    neck_mask = cv2.resize(neck_mask, (img_shape[0], img_shape[1]))


    return neck_mask,original_face_mask,original_neck_mask,chin_edge,full_mask
def get_maskgan_mask(img_path,net=None, cp='79999_iter.pth'):
    if isinstance(img_path, str):
        img= cv2.imread(img_path)
    else:
        img=img_path

    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        if isinstance(img_path,str):
            image = Image.open(img_path)
        else:
            image = Image.fromarray(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        neck_mask,other_mask,face_mask=vis_parsing_maps(parsing, stride=1,cloth_mask=False)

    return neck_mask


def get_full_mask(img_path,net=None, cp='79999_iter.pth'):
    if net == None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        # print(save_pth)
        # save_pth = osp.join(os.path.dirname(__file__),os.path.join('feature_extractor/face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        image = Image.open(img_path)
        image = image.resize((512, 512), Image.BILINEAR)
        img_input = to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        full_mask,full_mask_wo_hair = fullmask(parsing, stride=1)
    full_mask = cv2.resize(full_mask, (1024, 1024))
    full_mask_wo_hair = cv2.resize(full_mask_wo_hair, (1024, 1024))
    # cv2.imshow('i', img * ( full_mask_wo_hair // 255) )
    # cv2.waitKey(0)
    return full_mask,full_mask_wo_hair