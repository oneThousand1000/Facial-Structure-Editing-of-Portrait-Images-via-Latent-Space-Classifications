import numpy as np
import cv2
from scipy.spatial import Delaunay



'''
https://github.com/spmallick/learnopencv
'''

def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


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
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect =warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:(r[1] + r[3]), r[0]:(r[0] + r[2])] = img[r[1]:(r[1] + r[3]), r[0]:(r[0] + r[2])] * (1 - mask) + imgRect * mask





def get_tris(points,w,h,sample_num=4):


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

    points= np. concatenate([seeds,points],axis=0)
    delaunay = Delaunay(points)
    return delaunay,seeds


def warp(img_debug,img,points1_input,points2_input,debug=False):
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

    tris = delaunay.simplices
    if debug:
        can1 = img.copy()
        for tri in tris:
            x, y, z = tri
            cv2.line(can1, (points2[x][0], points2[x][1]), (points2[y][0], points2[y][1]), (255, 255, 255), 2)
            cv2.line(can1, (points2[y][0], points2[y][1]), (points2[z][0], points2[z][1]), (255, 255, 255), 2)
            cv2.line(can1, (points2[z][0], points2[z][1]), (points2[x][0], points2[x][1]), (255, 255, 255), 2)

        can2 = img_debug.copy()
        for tri in tris:
            x, y, z = tri
            cv2.line(can2, (points1[x][0],points1[x][1]), (points1[y][0], points1[y][1]), (255, 255, 255), 2)
            cv2.line(can2, (points1[y][0], points1[y][1]), (points1[z][0], points1[z][1]), (255, 255, 255),2)
            cv2.line(can2, (points1[z][0], points1[z][1]), (points1[x][0], points1[x][1]), (255, 255, 255), 2)


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