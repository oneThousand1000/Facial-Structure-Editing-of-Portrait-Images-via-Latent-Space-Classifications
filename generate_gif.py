import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from warp.warpper import warp_img
from classifier.src.feature_extractor.neck_mask_extractor import get_neck_blur_mask, get_parsingNet
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
        description='Edit image synthesis with given semantic boundary.') #
    parser.add_argument('-i', '--latent_path', type=str, default='F:/DoubleChin/datasets/ffhq_data/compare_stylespace/code/052932_wp.npy',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-o', '--output_dir', type=str,
                        default='./docs/results/gif',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('-m', '--image_path', type=str,
                        default='F:/DoubleChin/datasets/ffhq_data/double_chin_psi_0.8/origin/002405.jpg',
                        help='If specified, will load latent codes from given ')
    parser.add_argument('--boundary_path1', type=str,
                        default='./interface/boundaries/fine/all/boundary.npy',
                        help='Path to the semantic boundary. (required)')


    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('--boundary_begin_ratio', type=float, default=0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('--boundary_end_ratio', type=float, default=-6,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('--step_num', type=int, default=10,
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



    pbar=tqdm(total=args.step_num)

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    neckMaskNet = get_parsingNet()

    wps_latent = np.reshape(np.load(args.latent_path), (1, 18, 512))
    res_imgs_f=[]
    origin_img=cv2.imread('F:/DoubleChin/datasets/ffhq_data/compare_stylespace/origin/052932.jpg')
    mask = get_neck_blur_mask(img_path=origin_img, net=neckMaskNet, dilate=5)
    for step in range(args.step_num+1):
        pbar.update(1)
        edited_wps_latent_f = wps_latent + (args.boundary_end_ratio-args.boundary_begin_ratio)/args.step_num*step*boundary_f
        edited_output_f = model.easy_style_mixing(latent_codes=edited_wps_latent_f,
                                                style_range=range(6, 18),
                                                style_codes=wps_latent,
                                                mix_ratio=1.0, **kwargs)

        edited_img_f = edited_output_f['image'][0]#[:, :, ::-1]

        warpped_edited_img = warp_img(origin_img, edited_img_f, net=neckMaskNet, debug=False)
        res = warpped_edited_img * (mask / 255) + origin_img[:, :, ::-1] * (1 - mask / 255)
        res_imgs_f.append(res)
        #cv2.imwrite(save_path,res)


    # cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_bf.jpg'),np.concatenate(res_imgs_f,axis=1))
    # cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_bc.jpg'), np.concatenate(res_imgs_c, axis=1))
    #cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_temp.jpg'), np.concatenate(temp, axis=1))
    #res_imgs_f.reverse()
    print('save to ',os.path.join(args.output_dir, f'{image_name}.gif'))
    imageio.mimsave('F:/DoubleChin/datasets/ffhq_data/compare_stylespace/warp_res/ours_res.gif', res_imgs_f , 'GIF', duration=0.3)


if __name__ == '__main__':
    run()


