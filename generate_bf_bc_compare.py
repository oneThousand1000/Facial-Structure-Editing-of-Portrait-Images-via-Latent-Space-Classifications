import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from classifier.src.feature_extractor.neck_mask_extractor import get_neck_blur_mask,get_parsingNet,get_neck_mask
from warp.warpper import warp_img
import glob
import time
from CHINGER_inverter import StyleGAN2Inverter
'''
Data prepare:
For real images process, you should input `--data_dir PATH`,
put original real images in $PATH/origin, named `{name}.jpg`,
the corresponding wp latent code should be put in $PATH/code,
named `{name}_wp.npy`.
'''
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('-i', '--data_dir', type=str, default='F:/DoubleChin/landmarks/3DDFA/sample_paper',
                        help='If specified, will load latent codes from given ')

    parser.add_argument('--boundary_path1', type=str,
                        default='./interface/boundaries/fine/all/boundary.npy',
                        help='Path to the semantic boundary. (required)')
    parser.add_argument( '--boundary_path2', type=str,
                        default='./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w/boundary.npy',
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('--boundary_path3', type=str,
                        default='./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w_c_pose/boundary.npy',
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('--boundary_path4', type=str,
                        default='./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_pose/boundary.npy',
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('--boundary_init_ratio', type=float, default=-4.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')

    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')

    parser.add_argument("--compare_with_diffuse", type=str2bool, nargs='?',
                        const=False, default=False,
                        help="diffuse until no double chin.")
    return parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def diffuse(init_code,target,mask,inverter):
    kwargs = {'latent_space_type': 'wp'}
    target = target[:, :, ::-1]
    code, viz_result = inverter.easy_mask_diffuse(target=target,
                                                  init_code=init_code,
                                                  mask=mask,
                                                  **kwargs)

    viz_result = viz_result[:, :, ::-1]
    return viz_result


def run():
    model_name='stylegan2_ffhq'
    args = parse_args()
    latent_space_type = args.latent_space_type

    assert os.path.exists(args.data_dir),f'data_dir {args.data_dir} dose not exist!'
    origin_img_dir=os.path.join(args.data_dir, 'origin')
    code_dir=os.path.join(args.data_dir, 'code')
    res_path = os.path.join(args.data_dir, 'res')

    mkdir(res_path)


    model = StyleGAN2Generator(model_name, logger=None)

    kwargs = {'latent_space_type': latent_space_type}




    boundary1 = np.load(args.boundary_path1)
    boundary2 = np.load(args.boundary_path2)
    boundary3 = np.load(args.boundary_path3)
    boundary4 = np.load(args.boundary_path4)

    print(f'Load latent codes and images from `{args.data_dir}`.')
    latent_codes = []
    origin_img_list = []
    for img in glob.glob(os.path.join(origin_img_dir, '*.jpg')):
        name = os.path.basename(img)[:6]
        code_path = os.path.join(code_dir,f'{name}_wp.npy')
        if os.path.exists(code_path):
            latent_codes.append(code_path)
            origin_img_list.append(img)
    total_num = len(latent_codes)

    print(f'Processing {total_num} samples.')


    pbar=tqdm(total=total_num)

    for img_index in range(total_num):
        pbar.update(1)
        image_name = os.path.splitext(os.path.basename(origin_img_list[img_index]))[0]
        wps_latent = np.reshape(np.load(latent_codes[img_index]), (1, 18, 512))
        # if int(image_name)<200:
        #    origin_img = model.easy_synthesize(latent_codes=wps_latent,**kwargs)['image'][0][:, :, ::-1]
        #    cv2.imwrite(os.path.join(origin_img_dir, image_name + '.jpg'), origin_img)
        edited_wps_latent_bf = wps_latent + args.boundary_init_ratio * boundary1
        edited_wps_latent_bc = wps_latent + args.boundary_init_ratio * boundary2
        edited_wps_latent_bc_c_pose = wps_latent + args.boundary_init_ratio * boundary3
        edited_wps_latent_pose = wps_latent - args.boundary_init_ratio * boundary4



        bf_img = model.easy_style_mixing(latent_codes=edited_wps_latent_bf,
                                                style_range=range(7, 18),
                                                style_codes=wps_latent,
                                                mix_ratio=1.0, **kwargs)['image'][0][:, :, ::-1]
        bc_img = model.easy_style_mixing(latent_codes=edited_wps_latent_bc,
                                         style_range=range(7, 18),
                                         style_codes=wps_latent,
                                         mix_ratio=1.0, **kwargs)['image'][0][:, :, ::-1]

        bc_c_pose_img = model.easy_style_mixing(latent_codes=edited_wps_latent_bc_c_pose,
                                         style_range=range(7, 18),
                                         style_codes=wps_latent,
                                         mix_ratio=1.0, **kwargs)['image'][0][:, :, ::-1]
        bc_c_pose = model.easy_style_mixing(latent_codes=edited_wps_latent_pose,
                                                style_range=range(7, 18),
                                                style_codes=wps_latent,
                                                mix_ratio=1.0, **kwargs)['image'][0][:, :, ::-1]

        cv2.imwrite(os.path.join(res_path,image_name+'.jpg'),np.concatenate([cv2.imread(origin_img_list[img_index]),bc_img,bf_img],axis=1))
        cv2.imwrite(os.path.join(res_path,image_name+'_.jpg'),np.concatenate([bc_c_pose,bc_img,bc_c_pose_img],axis=1))




if __name__ == '__main__':
    run()


