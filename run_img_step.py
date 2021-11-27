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
    parser.add_argument('-i', '--latent_path', type=str, default='F:/DoubleChin/datasets/ffhq_data/real_img_author/code/04660_wp.npy', #
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-o', '--output_dir', type=str,
                        default='F:/CHINGER/docs/bishe/good',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-m', '--image_path', type=str,
                        default='F:/DoubleChin/datasets/ffhq_data/real_img_author/origin/04660.jpg',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-b', '--boundary_path', type=str,
                        default='./interface/boundaries/fine/all',
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')

    parser.add_argument('--step_num', type=int, default=5,
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
    model = StyleGAN2Generator(model_name, logger=None,randomize_noise=False)
    kwargs = {'latent_space_type': latent_space_type}



    print(f'Preparing boundary.')
    boundary_path = os.path.join(args.boundary_path, 'boundary.npy')
    intercept_path = os.path.join(args.boundary_path, 'intercept.npy')
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{boundary_path}` does not exist!')
    boundary = np.load(boundary_path)


    neckMaskNet = get_parsingNet()

    pbar=tqdm(total=args.step_num)

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]


    wps_latent = np.reshape(np.load(args.latent_path), (1, 18, 512))
    origin_img = cv2.imread(args.image_path)
    mask = get_neck_blur_mask(img_path=origin_img, net=neckMaskNet, dilate=5)
    res_imgs=[]
    distance=-10
    for step in range(args.step_num+1):
        pbar.update(1)
        # distance = np.abs(
        #     (np.sum(boundary * wps_latent, axis=2, keepdims=True) + intercept) / np.linalg.norm(boundary, axis=2,
        #                                                                                         keepdims=True)) * (-1.8)
        print(distance/args.step_num*step)
        edited_wps_latent = wps_latent + distance/args.step_num*step*boundary

        edited_output = model.easy_style_mixing(latent_codes=edited_wps_latent,
                                                style_range=range(6, 18),
                                                style_codes=wps_latent,
                                                mix_ratio=1.0, **kwargs)

        edited_img = edited_output['image'][0][:, :, ::-1]

        warpped_edited_img = warp_img(origin_img, edited_img, net=neckMaskNet, debug=False)
        res= warpped_edited_img * (mask / 255) + origin_img * (1 - mask / 255)
        res_imgs.append(res)


    cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_step{step}_fixed-test.jpg'),np.concatenate(res_imgs,axis=1))


if __name__ == '__main__':
    run()


