import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from classifier.src.feature_extractor.neck_mask_extractor import get_neck_blur_mask, get_parsingNet
from warp.warpper import warp_img
import glob
import time
'''
Data prepare:
For real images process, you should input `--data_dir PATH`,
put original real images in $PATH/origin, named `{name}.jpg`,
the corresponding wp latent code should be put in $PATH/code,
named `{name}_wp.npy`.
'''


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('-i', '--data_dir', type=str, default='F:/DoubleChin/datasets/ffhq_data/fake_img_add',
                        help='If specified, will load latent codes from given ')

    parser.add_argument('-b', '--boundary_path', type=str,
                        default='./interface/boundaries/fine/all_',
                        help='Path to the semantic boundary. (required)')


    parser.add_argument('--boundary_init_ratio', type=float, default=10,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')

    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')

    return parser.parse_args()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def diffuse(init_code, target, mask, inverter):
    kwargs = {'latent_space_type': 'wp'}
    target = target[:, :, ::-1]
    code, viz_result = inverter.easy_mask_diffuse(target=target,
                                                  init_code=init_code,
                                                  mask=mask,
                                                  **kwargs)

    viz_result = viz_result[:, :, ::-1]
    return viz_result


def run():
    model_name = 'stylegan2_ffhq'
    args = parse_args()
    latent_space_type = args.latent_space_type

    assert os.path.exists(args.data_dir), f'data_dir {args.data_dir} dose not exist!'
    origin_img_dir = os.path.join(args.data_dir, 'origin')
    code_dir = os.path.join(args.data_dir, 'code')
    res_dir = os.path.join(args.data_dir, 'warp_res')
    temp_dir = os.path.join(args.data_dir, 'temp')
    assert os.path.exists(origin_img_dir), f'{origin_img_dir} dose not exist!'
    assert os.path.exists(code_dir), f'data_dir {code_dir} dose not exist!'
    mkdir(res_dir)
    mkdir(temp_dir)

    print(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger=None)

    kwargs = {'latent_space_type': latent_space_type}

    print(f'Preparing boundary.')
    boundary_path=os.path.join(args.boundary_path,'boundary.npy')
    intercept_path = os.path.join(args.boundary_path, 'intercept.npy')
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{boundary_path}` does not exist!')
    if not os.path.isfile(intercept_path):
        raise ValueError(f'Boundary `{intercept_path}` does not exist!')
    boundary = np.load(boundary_path)

    print(f'Load latent codes and images from `{args.data_dir}`.')
    latent_codes = []
    origin_img_list = []
    for img in glob.glob(os.path.join(origin_img_dir, '*.jpg'))+glob.glob(os.path.join(origin_img_dir, '*.png')):
        name = os.path.basename(img)[:-4]
        code_path = os.path.join(code_dir, f'{name}_wp.npy')
        if os.path.exists(code_path):
            latent_codes.append(code_path)
            origin_img_list.append(img)
    total_num = len(latent_codes)

    print(f'Processing {total_num} samples.')

    neckMaskNet = get_parsingNet()

    pbar = tqdm(total=total_num)
    times = []
    img_count = 0

    for img_index in range(total_num):
        pbar.update(1)
        image_name = os.path.splitext(os.path.basename(origin_img_list[img_index]))[0]
        if os.path.exists(os.path.join(res_dir, f'{image_name}.jpg')):
            continue

        try:

            wps_latent = np.reshape(np.load(latent_codes[img_index]), (1, 18, 512))
            origin_img = cv2.imread(origin_img_list[img_index])
            distance=args.boundary_init_ratio

            edited_wps_latent = wps_latent + distance * boundary
            edited_output = model.easy_style_mixing(latent_codes=edited_wps_latent,
                                                    style_range=range(6, 18),
                                                    style_codes=wps_latent,
                                                    mix_ratio=1.0, **kwargs)
            #

            edited_img = edited_output['image'][0][:, :, ::-1]


            mask = get_neck_blur_mask(img_path=edited_img, net=neckMaskNet, dilate=5)

            warpped_edited_img = warp_img(origin_img, edited_img, net=neckMaskNet, debug=False)


            res = warpped_edited_img* (mask / 255) + origin_img *(1 - mask / 255)

            cv2.imwrite(os.path.join(res_dir, f'{image_name}.jpg'), res)
            cv2.imwrite(os.path.join(temp_dir, f'{image_name}.jpg'),
                        np.concatenate([origin_img,warpped_edited_img, res], axis=1))
            img_count += 1
        except:
            pass

    print(times)
    times = np.array(times)
    print(times.shape)


if __name__ == '__main__':
    run()
