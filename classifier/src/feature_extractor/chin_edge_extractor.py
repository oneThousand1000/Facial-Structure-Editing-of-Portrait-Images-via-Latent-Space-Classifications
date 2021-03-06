import sys
sys.path.append('./')

from .chin_edge_generator  import face_alignment
import cv2
import numpy as np
def get_chin_point(img):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    preds = fa.get_landmarks_from_image(img)


    return (preds[0][8][1], preds[0][8][1])





def chin_edge_extractor(img,debug=False):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    preds = fa.get_landmarks_from_image(img)
    preds = preds[0]

    #print(preds.shape)
    can= np.zeros((img.shape[0],img.shape[1],3))
    #
    for i in range(0, 17):
        point = preds[i]
        cv2.putText(img,str(i),(point[0],point[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),5)
        if (i != 0):
            cv2.line(can, (preds[i - 1][0], preds[i - 1][1]), (point[0], point[1]), (255, 255, 255), 5)
            cv2.line(img, (preds[i - 1][0], preds[i - 1][1]), (point[0], point[1]), (255, 255, 255), 5)

    if debug:
        cv2.imshow('i',can)
        cv2.waitKey(0)
    return can
def warpping_area(img):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    preds = fa.get_landmarks_from_image(img)

    can= np.ones((img.shape[0],img.shape[1],3))*255

    can[:int(max(preds[0][0][1],preds[0][16][1])),:,:]=0


    can[int(min(preds[0][7][1],preds[0][10][1])):,:,:]=0

    can=can.astype(np.uint8)
    return can,int(preds[0][8][1])

def chin_mask_extractor(img,debug=False):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    preds = fa.get_landmarks_from_image(img)
    preds = preds[0]
    can = np.zeros_like(img)

    cv2.fillPoly(can, np.array([preds[:17]],dtype=np.int), 255)

    can[:,:,1]=can[:,:,0]
    can[:, :, 2]=can[:,:,0]
    if debug:
        cv2.imshow('i', can//255*img)
        cv2.waitKey(0)
    return can,min((preds[1][1]),(preds[15][1]))

