from warp.utils import warp
import cv2
import numpy as np
from classifier.src.feature_extractor.neck_mask_extractor import get_warpping_mask,get_face_mask,get_parsingNet


def get_point(mask1,mask2,point_pair_num):

    index1 = np.where(mask1!= 0)
    index2 = np.where(mask2!= 0)
    high = max(min(index1[0]), min(index2[0]))
    low = min(max(index1[0]), max(index2[0]))
    points1_left = []
    points2_left = []
    points1_right = []
    points2_right = []
    # coefs=[]
    for y_cor in np.linspace(high, low, point_pair_num + 1).astype(np.uint32)[:-1]:
        choose = np.where(mask1[y_cor,:, :] != 0)
        x_cor1_left = min(choose[0])
        x_cor1_right = max(choose[0])
        choose = np.where(mask2[y_cor, :, :] != 0)
        x_cor2_left = min(choose[0])
        x_cor2_right = max(choose[0])
        points1_left.append([x_cor1_left, y_cor])
        points2_left.append([x_cor2_left, y_cor])

        points1_right.append([x_cor1_right, y_cor])
        points2_right.append([x_cor2_right, y_cor])

    points1_left = points1_left[::-1]
    points2_left = points2_left[::-1]


    return np.array(points1_left),np.array(points1_right),np.array(points2_left),np.array(points2_right)


def warp_img(img1,img2,net,debug=False):
    if net ==None:
        net = get_parsingNet()

    face_mask1,neck_mask1 = get_face_mask(img1, net)
    face_mask2,neck_mask2 = get_face_mask(img2, net)

    warpping_mask,chin_point= get_warpping_mask(img=img1)

    point_pair_num = 5

    face_mask1 = face_mask1 * (warpping_mask // 255)
    face_mask2 = face_mask2 * (warpping_mask // 255)
    #

    neck_mask1[:chin_point,:,:]=0
    neck_mask2[:chin_point, :, :]=0



    face_points1_left,face_points1_right,face_points2_left,face_points2_right= get_point(mask1=face_mask1,mask2=face_mask2,point_pair_num=point_pair_num)


    _,_,neck_points2_left,neck_points2_right = get_point(mask1=neck_mask1,
                                                         mask2=neck_mask2,
                                                         point_pair_num=point_pair_num
                                                         )

    face_points2_left = face_points1_left+(face_points2_left-face_points1_left)*1.2
    face_points2_right = face_points1_right + (face_points2_right - face_points1_right) * 1.2


    points1=(np.concatenate([face_points1_left,neck_points2_left,neck_points2_right,face_points1_right],axis=0)).astype(np.uint32)
    points2 = (np.concatenate([face_points2_left, neck_points2_left, neck_points2_right, face_points2_right], axis=0)).astype(np.uint32)

    if debug:
        can1 = img1.copy()
        for i  in range(points1.shape[0]):
            cv2.putText(can1, str(i), (points1[i][0], points1[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
            cv2.circle(can1, (points1[i][0], points1[i][1]), radius=25, color=(255, 0, 0))

        can2 = img2.copy()
        for i  in range(points2.shape[0]):
            cv2.putText(can2, str(i), (points2[i][0], points2[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
            cv2.circle(can2, (points2[i][0], points2[i][1]), radius=25, color=(255, 0, 0))


    if debug:
        warpped_img, debug_img = warp(img1,img2, points1, points2, debug=debug)
        return warpped_img, np.concatenate([can1,can2,debug_img,face_mask1,neck_mask1,face_mask2,neck_mask2],axis=1)
    else:
        warpped_img = warp(img1,img2, points1, points2, debug=debug)
        return warpped_img

