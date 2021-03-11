import numpy as np
import os
from skimage.io import imread
from skimage.transform import rescale
import argparse
import ast
from .api import PRN

def get_PRNet_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prn = PRN(is_dlib=True,prefix=os.path.dirname(os.path.realpath(__file__)))
    return prn

def PRNet_landmarks_extractor(img,prn):


    # ---- init PRN


    if isinstance(img,str):
        image = imread(img)
    else:
        image=img

    [h, w, c] = image.shape
    if c > 3:
        image = image[:, :, :3]

    # the core: regress position map
    max_size = max(image.shape[0], image.shape[1])
    if max_size > 1000:
        image = rescale(image, 1000. / max_size)
        image = (image * 255).astype(np.uint8)
    pos = prn.process(image)
    kpt = prn.get_landmarks(pos)
    kpt = np.round(kpt).astype(np.int32)
    kpt=kpt[:,:2]
    return kpt
    # # ------------- load data
    # image_folder = args.inputDir
    # save_folder = args.outputDir
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    #
    # types = ('*.jpg', '*.png')
    # image_path_list = []
    # for files in types:
    #     image_path_list.extend(glob(os.path.join(image_folder, files)))
    # total_num = len(image_path_list)

    # for i, image_path in enumerate(image_path_list):
    #     print(image_path)
    #     name = os.path.basename(image_path[:-4])
    #
    #     # read image
    #     image = imread(image_path)


        # image = image / 255.
        # if pos is None:
        #     continue
        #
        # if args.is3d or args.isMat or args.isPose or args.isShow:
        #     # 3D vertices
        #     vertices = prn.get_vertices(pos)
        #     if args.isFront:
        #         save_vertices = frontalize(vertices)
        #     else:
        #         save_vertices = vertices.copy()
        #     save_vertices[:, 1] = h - 1 - save_vertices[:, 1]
        #
        # if args.isImage:
        #     imsave(os.path.join(save_folder, name + '.jpg'), image)
        #
        # if args.is3d:
        #     # corresponding colors
        #     colors = prn.get_colors(image, vertices)
        #
        #     if args.isTexture:
        #         if args.texture_size != 256:
        #             pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range=True)
        #         else:
        #             pos_interpolated = pos.copy()
        #         texture = cv2.remap(image, pos_interpolated[:, :, :2].astype(np.float32), None,
        #                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        #         if args.isMask:
        #             vertices_vis = get_visibility(vertices, prn.triangles, h, w)
        #             uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
        #             uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range=True)
        #             texture = texture * uv_mask[:, :, np.newaxis]
        #         write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture,
        #                                prn.uv_coords / prn.resolution_op)  # save 3d face with texture(can open with meshlab)
        #     else:
        #         write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles,
        #                               colors)  # save 3d face(can open with meshlab)
        #
        # if args.isDepth:
        #     depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
        #     depth = get_depth_image(vertices, prn.triangles, h, w)
        #     imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
        #     sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth': depth})
        #
        # if args.isMat:
        #     sio.savemat(os.path.join(save_folder, name + '_mesh.mat'),
        #                 {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})
        #
        #
        # imsave(os.path.join(save_folder, name + '_kpt.jpg'), plot_kpt(image, kpt))
        # if args.isShow:
        #     # ---------- Plot
        #     image_pose = plot_pose_box(image, camera_matrix, kpt)
        #     cv2.imshow('sparse alignment', plot_kpt(image, kpt))
        #
        #     cv2.imshow('dense alignment', plot_vertices(image, vertices))
        #     cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
        #     cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/test1/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/test1_res', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='1', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=True, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=True, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=True, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=True, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=True, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=True, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=True, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=True, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    main(parser.parse_args())
