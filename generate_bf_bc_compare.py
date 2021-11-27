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
    parser.add_argument('-i', '--data_dir', type=str, default='F:/DoubleChin/datasets/ffhq_data/real_img_author/code',
                        help='If specified, will load latent codes from given ')

    parser.add_argument( '--boundary_path', type=str,
                        default='./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w/boundary.npy',
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




    boundary = np.load(args.boundary_path)


    for img in glob.glob('F:/DoubleChin/comparision/samples3/origin/*.jpg'):
        image_name = os.path.splitext(os.path.basename(img))[0]
        wps_latent = np.reshape(np.load(os.path.join(args.data_dir,f'{image_name}_wp.npy')), (1, 18, 512))

        edited_wps_latent_bc = wps_latent + args.boundary_init_ratio * boundary




        bc_img = model.easy_style_mixing(latent_codes=edited_wps_latent_bc,
                                         style_range=range(7, 18),
                                         style_codes=wps_latent,
                                         mix_ratio=1.0, **kwargs)['image'][0][:, :, ::-1]



        cv2.imwrite(os.path.join('F:/DoubleChin/comparision/samples3/interfacegan',image_name+'.jpg'),bc_img)




if __name__ == '__main__':
    run()


