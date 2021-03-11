from warp.utils import warp,data_load
import cv2
import numpy as np
from classifier.src.feature_extractor.neck_mask_extractor import get_warpping_mask,get_face_mask,get_parsingNet
import torch

img_list=data_load()
net=get_parsingNet()
for img_index in range(len(img_list)):
    img = cv2.imread(img_list[img_index][0])
    img1 = cv2.imread(img_list[img_index][1])



    face_mask1=get_face_mask(img,net)
    face_mask2 = get_face_mask(img1, net)
    # face_edge1 = cv2.Canny(face_mask1, 0, 0)
    # face_edge1 = np.stack((face_edge1, face_edge1, face_edge1), axis=2)
    # face_edge2 = cv2.Canny(face_mask2, 0, 0)
    # face_edge2 = np.stack((face_edge2, face_edge2, face_edge2), axis=2)

    mask, _ = get_warpping_mask(img=img)

    # cv2.imshow('i', mask)
    # cv2.waitKey(0)

    point_pair_num=7

    face_mask1 = face_mask1* (mask // 255)
    face_mask2 = face_mask2 * (mask // 255)

    left1= np.where(face_mask1[:,:mask.shape[0]//2,:]!=0)
    left2 = np.where(face_mask2[:, :mask.shape[0] // 2, :] != 0)
    left_low = min(max(left1[0]),max(left2[0]))
    left_high =max(min(left1[0]),min(left2[0]))
    points1=[]
    points2 = []
    #coefs=[]
    for index in range(1,point_pair_num+1):
        #coefs.append(1/(point_pair_num)*index)
        y_cor=left_low+int((left_high-left_low)/(point_pair_num+1)*index)
        choose=np.where(face_mask1[y_cor,:face_mask1.shape[0]//2,:]!=0)
        x_cor1 = min(choose[0])
        choose = np.where(face_mask2[y_cor, :face_mask2.shape[0] // 2, :] != 0)
        x_cor2 = min(choose[0])
        points1.append([x_cor1,y_cor])
        points2.append([x_cor2,y_cor])
    points1=points1[::-1]
    points2 = points2[::-1]
    #coefs= coefs[::-1]
    right1 = np.where(face_mask1[:,  mask.shape[0] // 2:, :] != 0)
    right2 = np.where(face_mask2[:,  mask.shape[0] // 2:, :] != 0)
    right_low = min(max(right1[0]), max(right2[0]))
    right_high = max(min(right1[0]), min(right2[0]))

    for index in range(1,point_pair_num+1):
        #coefs.append(1 / (point_pair_num ) *index)
        y_cor = right_low + int((right_high - right_low) / (point_pair_num+1) * index)
        choose = np.where(face_mask1[y_cor, face_mask1.shape[0] // 2:, :] != 0)
        x_cor1 = max(choose[0])
        choose = np.where(face_mask2[y_cor, face_mask2.shape[0] // 2:, :] != 0)
        x_cor2 = max(choose[0])
        points1.append([x_cor1+face_mask1.shape[0] // 2, y_cor])
        points2.append([x_cor2+face_mask2.shape[0] // 2, y_cor])
    can1 = img.copy()
    for i, point in enumerate(points1):
        cv2.putText(can1, str(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
        cv2.circle(can1, (point[0], point[1]), radius=25, color=(255, 0, 0))

    # cv2.imshow('i', cv2.resize(can1, (512, 512)))
    # cv2.waitKey(0)

    can2 = img1.copy()
    for i, point in enumerate(points2):
        cv2.putText(can2, str(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
        cv2.circle(can2, (point[0], point[1]), radius=25, color=(255, 0, 0))

    # cv2.imshow('i', cv2.resize(can2, (512, 512)))
    # cv2.waitKey(0)
    points1 = np.array(points1)
    points2 = np.array(points2)
    coefs=[1.5,2.0,2.0,1.5,0.8,0.4,0.2,0.2,0.4,0.8,1.5,2.0,2.0,1.5]
    coefs = np.array(coefs)
    coefs=np.reshape(coefs, (-1, 1))

    points2=points1+(points2-points1)*coefs
    points2=points2.astype(np.int32)
    # coefs=np.array(coefs)
    # coefs = np.reshape(coefs,(-1,1))
    # print(coefs)

    # points=np.array(points)
    # translations = np.zeros((8, 2), int)
    # translations[0][0] = translations[1][0] = translations[2][0] = translations[3][0] = 80
    # translations[4][0] = translations[5][0] = translations[6][0] = translations[7][0] = -80
    # points2 = (points2-points1)*coefs+points1
    # points2 = points2.astype(np.int32)
    warpped_img,debug_img = warp(img1, points1, points2,debug=True)

    mask_dilate_blur = cv2.resize(cv2.imread(img_list[img_index][2]),(1024,1024))
    # mask = (neck_mask > 0).astype(np.uint8) * 255
    # mask_dilate = cv2.dilate(mask, kernel=np.ones((30, 30), np.uint8))
    # mask_dilate_blur = cv2.blur(mask_dilate, ksize=(35, 35))
    # mask_dilate_blur = mask + (255 - mask) // 255 * mask_dilate_blur

    # cv2.imshow('i', cv2.resize(warpped_img, (256, 256)))
    # cv2.waitKey(0)
    # res = np.concatenate([(img * (mask // 255)).astype(np.uint8), (warpped_img * (mask // 255)).astype(np.uint8)],
    #                      axis=1)
    # cv2.imshow('i', cv2.resize((res).astype(np.uint8), (1024, 512)))
    # cv2.waitKey(0)
    # #
    res = img * (1 - mask_dilate_blur / 255) + warpped_img * (mask_dilate_blur / 255)  # np.concatenate([(img * (mask // 255)).astype(np.uint8), (warpped_img * (mask // 255)).astype(np.uint8)],axis=1)
    res_origin = img * (1 - mask_dilate_blur / 255) + img1 * (mask_dilate_blur / 255)
    res=np.concatenate([can1,can2,debug_img,res,res_origin],axis=1)
    cv2.imwrite(f'F:/DoubleChin/datasets/ffhq_data/double_chin_pair/warp_debug/{img_index}.jpg',res)
    # cv2.imshow('i', cv2.resize((res).astype(np.uint8), (1024, 512)))
    # cv2.waitKey(0)


