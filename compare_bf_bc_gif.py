import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from classifier.src.feature_extractor.neck_mask_extractor import get_neck_blur_mask,get_parsingNet,get_neck_mask
from warp.warpper import warp_img
import glob

from CHINGER_inverter import StyleGAN2Inverter
'''
Data prepare:
For real images process, you should input `--data_dir PATH`,
put original real images in $PATH/origin, named `{name}.jpg`,
the corresponding wp latent code should be put in $PATH/code,
named `{name}_wp.npy`.
'''
import imageio

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
    parser.add_argument('-i', '--latent_path', type=str, default='F:/DoubleChin/datasets/ffhq_data/double_chin_pair_0.8_/codes/000335_wp.npy',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-o', '--output_dir', type=str,
                        default='./docs/results/gif',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-m', '--image_path', type=str,
                        default='F:/DoubleChin/datasets/ffhq_data/double_chin_pair_0.8_/images/000335_w_doublechin.jpg',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('--boundary_path1', type=str,
                        default='./interface/boundaries/fine/all/boundary.npy',
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('--boundary_path2', type=str,
                        default='./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w/boundary.npy',
                        help='Path to the semantic boundary. (required)')


    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('--boundary_begin_ratio', type=float, default=0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('--boundary_end_ratio', type=float, default=-5.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('--step_num', type=int, default=30,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    return parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)



def run():
    model_name='stylegan2_ffhq'
    args = parse_args()
    latent_space_type = args.latent_space_type

    assert os.path.exists(args.output_dir),f'data_dir {args.output_dir} dose not exist!'


    print(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger=None)
    kwargs = {'latent_space_type': latent_space_type}



    print(f'Preparing boundary.')

    boundary_f = np.load(args.boundary_path1)
    boundary_c = np.load(args.boundary_path2)



    pbar=tqdm(total=args.step_num)

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]


    wps_latent = np.reshape(np.load(args.latent_path), (1, 18, 512))
    res_imgs_f=[]
    res_imgs_c = []
    for step in range(args.step_num+1):
        pbar.update(1)
        edited_wps_latent_f = wps_latent + (args.boundary_end_ratio-args.boundary_begin_ratio)/args.step_num*step*boundary_f
        edited_output_f = model.easy_style_mixing(latent_codes=edited_wps_latent_f,
                                                style_range=range(6, 18),
                                                style_codes=wps_latent,
                                                mix_ratio=1.0, **kwargs)

        edited_img_f = edited_output_f['image'][0]#[:, :, ::-1]

        edited_wps_latent_c = wps_latent + (
                    args.boundary_end_ratio - args.boundary_begin_ratio) / args.step_num * step * boundary_c
        edited_output_c = model.easy_style_mixing(latent_codes=edited_wps_latent_c,
                                                  style_range=range(6, 18),
                                                  style_codes=wps_latent,
                                                  mix_ratio=1.0, **kwargs)

        edited_img_c = edited_output_c['image'][0]#[:, :, ::-1]


        #save_path=os.path.join(args.output_dir, f'{image_name}_step{step}.jpg')
        res_imgs_f.append(edited_img_f)
        res_imgs_c.append(edited_img_c)
        #cv2.imwrite(save_path,res)


    # cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_bf.jpg'),np.concatenate(res_imgs_f,axis=1))
    # cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_bc.jpg'), np.concatenate(res_imgs_c, axis=1))
    #cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_temp.jpg'), np.concatenate(temp, axis=1))
    print('save to ',os.path.join(args.output_dir, f'{image_name}_bf.gif'))
    imageio.mimsave(os.path.join(args.output_dir, f'{image_name}_bf.gif'), res_imgs_f, 'GIF', duration=0.3)
    imageio.mimsave(os.path.join(args.output_dir, f'{image_name}_bc.gif'), res_imgs_c, 'GIF', duration=0.3)


if __name__ == '__main__':
    run()


