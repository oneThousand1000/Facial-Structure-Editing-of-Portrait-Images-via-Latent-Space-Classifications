import os
import numpy as np
import  glob
import cv2
# from .pyDelaunay2D.delaunay2D import Delaunay2D
from scipy.spatial import Delaunay



'''

https://github.com/spmallick/learnopencv

'''

def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    #print(np.float32(srcTri).shape)
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    #img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    #warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect =warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:(r[1] + r[3]), r[0]:(r[0] + r[2])] = img[r[1]:(r[1] + r[3]), r[0]:(r[0] + r[2])] * (1 - mask) + imgRect * mask



def data_load():
    img_list=[]
    for img in glob.glob('F:/DoubleChin/datasets/ffhq_data/double_chin_pair/images/*_w_doublechin.jpg'):
        img1 = img
        img2 = img.replace('_w_doublechin','').replace('images','wo_double_chin')
        mask = img.replace('_w_doublechin', '').replace('images', 'mask_blur').replace('jpg', 'png')
        if os.path.exists(img2) and os.path.exists(mask):
            img_list.append((img1,img2,mask))
    return img_list

#
# tris=[
#     [0,1,8],
#     [1,8,9],
#     [1,9,10],
#     [1,10,11],
#     [1,11,12],
#     [1,12,13],
#     [1,13,14],
#     [1,14,15],
#     [1,2,15],
#     [2,15,3],
#     [3,4,15],
#     [4,14,15],
#     [4,13,14],
#     [4,12,13],
#     [4,5,12],
#     [5,11,12],
#     [5,6,11],
#     [6,11,10],
#     [6,9,10],
#     [6,8,9],
#     [6,7,8],
#     [0,7,8],
# ]


# def warp(img,points,translations):
#     #translations 6*2
#
#     img = img.astype(np.uint8)
#     imgMorph = np.zeros(img.shape, dtype=img.dtype)
#     w = img.shape[0]
#     h = img.shape[1]
#     print(translations)
#     points1 = [
#         (0, 0),
#         (int(w / 2), 0),
#         (w - 1, 0),
#         (w - 1, int(h / 2)),
#         (w - 1, h - 1),
#         (int(w / 2), h - 1),
#         (0, h - 1),
#         (0, int(h / 2)),
#         (points[0][0], points[0][1]),
#         (points[1][0], points[1][1]),
#         (points[2][0], points[2][1]),
#         (points[3][0], points[3][1]),
#         (points[4][0], points[4][1]),
#         (points[5][0], points[5][1]),
#         (points[6][0], points[6][1]),
#         (points[7][0], points[7][1])
#
#     ]
#     # can1=img.copy()
#     # for i,point in enumerate(points1):
#     #     cv2.putText(can1, str(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
#     #     cv2.circle(can1,(point[0],point[1]),radius=25,color=(255,0,0))
#     #
#     # cv2.imshow('i', cv2.resize(can1, (512, 512)))
#     # cv2.waitKey(0)
#     # print('===================================')
#     points2 = [
#         (0, 0),
#         (int(w / 2), 0),
#         (w - 1, 0),
#         (w - 1, int(h / 2)),
#         (w - 1, h - 1),
#         (int(w / 2), h - 1),
#         (0, h - 1),
#         (0, int(h / 2)),
#         (points[0][0] + translations[0][0], points[0][1]+ translations[0][1]),
#         (points[1][0] + translations[1][0], points[1][1]+ translations[1][1]),
#         (points[2][0] + translations[2][0], points[2][1]+ translations[2][1]),
#         (points[3][0]+ translations[3][0], points[3][1]+ translations[3][1]),
#         (points[4][0]+ translations[4][0], points[4][1]+ translations[4][1]),
#         (points[5][0]+ translations[5][0], points[5][1]+ translations[5][1]),
#         (points[6][0] + translations[6][0], points[6][1] + translations[6][1]),
#         (points[7][0] + translations[7][0], points[7][1] + translations[7][1])
#     ]
#     # can2 = img.copy()
#     # for point in points1:
#     #     cv2.circle(can2, (point[0], point[1]), radius=25, color=(255, 0, 0))
#     #
#     # cv2.imshow('i', cv2.resize(can2, (512, 512)))
#     # cv2.waitKey(0)
#     # print(points2)
#     # print('===================================')
#     points = []
#
#     alpha = 0.5
#     for i in range(0, len(points1)):
#         x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
#         y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
#         points.append((x, y))
#
#
#     # Read triangles from tri.txt
#     for tri in tris:
#         x, y, z = tri
#
#         t1 = [points1[x], points1[y], points1[z]]
#         t2 = [points2[x], points2[y], points2[z]]
#         t = [points[x], points[y], points[z]]
#
#         # Morph one triangle at a time.
#         morphTriangle(img, img.copy(), imgMorph, t1, t2, t, alpha)
#
#     return imgMorph

def get_tris(points,w,h,sample_num=4):
    # #print("points_pair_num:",points_pair_num)
    # tris = []
    # tris.append([0,1,7])
    # tris.append([1,2,3])
    # tris.append([4,5,8+points_pair_num])
    # tris.append([5,6,7+points_pair_num])
    # tris.append([5,8+points_pair_num, 7 + points_pair_num])
    # tris.append([1, 8 + points_pair_num, 7 + points_pair_num])
    # for i in range(7,7+points_pair_num):
    #     tris.append([i,i+1,1])
    #     tris.append([i, i + 1, 6])
    # for i in range(8 + points_pair_num,8 + points_pair_num*2):
    #     if i!= 8 + points_pair_num*2-1:
    #         tris.append([i, i + 1, 1])
    #         tris.append([i, i + 1, 4])
    #     else:
    #         tris.append([i, 3, 1])
    #         tris.append([i, 3, 4])
    # assert len(tris)==6+points_pair_num*4
    # return tris

    seeds = [
            (0, 0),
            (int(w / 2), 0),
            (w - 1, 0),
            (w - 1, int(h / 2)),
            (w - 1, h - 1),
            (int(w / 2), h - 1),
            (0, h - 1),
            (0, int(h / 2))

        ]

    # for i in range(1,sample_num):
    #     for j in range(1,sample_num):
    #         seeds.append((w/sample_num*i,h/sample_num*j))

    '''
    reference: 
        https://github.com/jmespadero/pyDelaunay2D
    '''
    seeds.append((w / 2, h / sample_num ))
    for i in range(1,sample_num*2):
        if i not in [3,4,5]:
            seeds.append((w / sample_num / 2 * i, h / sample_num / 2))
            seeds.append((w / sample_num / 2 * i, h / sample_num / 2 * (sample_num * 2 - 1)))
    for i in range(2,sample_num*2-1):
        seeds.append((w/sample_num/2,h/sample_num/2*i))
        seeds.append((w / sample_num / 2 *(sample_num*2-1), h / sample_num / 2*i))


    seeds = np.array(seeds)

    # dt = Delaunay2D()
    #
    # for s in seeds:
    #     dt.addPoint(s)
    #
    #
    # for s in points:
    #     dt.addPoint(s)
    #
    #
    # seeds= seeds.astype(np.uint32)
    points= np. concatenate([seeds,points],axis=0)
    delaunay = Delaunay(points)
    return delaunay,seeds
    #return dt.exportTriangles(),seeds


def warp(img_debug,img,points1_input,points2_input,debug=False):
    #translations 6*2
    assert points1_input.shape[0]==points2_input.shape[0]
    img = img.astype(np.uint8)
    imgMorph = np.zeros(img.shape, dtype=img.dtype)
    w = img.shape[0]
    h = img.shape[1]


    delaunay,random_points = get_tris(points1_input, w, h)

    points1=(np.concatenate([random_points,points1_input],axis=0)).astype(np.uint32)
    points2 = (np.concatenate([random_points, points2_input], axis=0)).astype(np.uint32)

    alpha = 0.5
    points =( (1 - alpha) * points1  + alpha * points2).astype(np.uint32)

    #print(tris.simplices.shape)
    tris = delaunay.simplices
    if debug:
        can1 = img.copy()
        for tri in tris:
            x, y, z = tri
            cv2.line(can1, (points2[x][0], points2[x][1]), (points2[y][0], points2[y][1]), (255, 255, 255), 2)
            cv2.line(can1, (points2[y][0], points2[y][1]), (points2[z][0], points2[z][1]), (255, 255, 255), 2)
            cv2.line(can1, (points2[z][0], points2[z][1]), (points2[x][0], points2[x][1]), (255, 255, 255), 2)
            #     #     cv2.circle(can2, (point[0], point[1]), radius=25, color=(255, 0, 0))
        #     #
        # cv2.imshow('i', cv2.resize(can1, (512, 512)))
        # cv2.waitKey(0)
        can2 = img_debug.copy()
        for tri in tris:
            x, y, z = tri
            cv2.line(can2, (points1[x][0],points1[x][1]), (points1[y][0], points1[y][1]), (255, 255, 255), 2)
            cv2.line(can2, (points1[y][0], points1[y][1]), (points1[z][0], points1[z][1]), (255, 255, 255),2)
            cv2.line(can2, (points1[z][0], points1[z][1]), (points1[x][0], points1[x][1]), (255, 255, 255), 2)
            #     #     cv2.circle(can2, (point[0], point[1]), radius=25, color=(255, 0, 0))
        #     #
        # cv2.imshow('i', cv2.resize(can2, (512, 512)))
        # cv2.waitKey(0)

    for tri in tris:
        x, y, z = tri

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(img, img.copy(), imgMorph, t1, t2, t, alpha)

    if debug:
        return imgMorph, np.concatenate([can1,can2],axis=1)
    return imgMorph