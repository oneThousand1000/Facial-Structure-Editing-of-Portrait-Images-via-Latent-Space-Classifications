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
    parser.add_argument('-i', '--data_dir', type=str, default='F:/DoubleChin/datasets/ffhq_data/real_img_select',
                        help='If specified, will load latent codes from given ')

    parser.add_argument('-b', '--boundary_path', type=str,
                        default='./interface/boundaries/fine/all/boundary.npy',
                        help='Path to the semantic boundary. (required)')

    parser.add_argument('--boundary_init_ratio', type=float, default=-10.0,
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
    if not os.path.isfile(args.boundary_path):
        raise ValueError(f'Boundary `{args.boundary_path}` does not exist!')
    boundary = np.load(args.boundary_path)

    print(f'Load latent codes and images from `{args.data_dir}`.')
    latent_codes = []
    origin_img_list = []
    for img in glob.glob(os.path.join(origin_img_dir, '*.jpg'))+glob.glob(os.path.join(origin_img_dir, '*.png')):
        name = os.path.basename(img)[:6]
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
            start_edit = time.clock()

            wps_latent = np.reshape(np.load(latent_codes[img_index]), (1, 18, 512))
            origin_img = cv2.imread(origin_img_list[img_index])
            edited_wps_latent = wps_latent + args.boundary_init_ratio * boundary
            origin_encode = model.easy_synthesize(wps_latent,
                                                  **kwargs)
            origin_encode = origin_encode['image'][0][:, :, ::-1]
            edited_output = model.easy_style_mixing(latent_codes=edited_wps_latent,
                                                    style_range=range(6, 18),
                                                    style_codes=wps_latent,
                                                    mix_ratio=1.0, **kwargs)
            #

            edited_img = edited_output['image'][0][:, :, ::-1]

            time_edit = (time.clock() - start_edit)

            # edited_img2 = edited_output2['image'][0][:, :, ::-1]
            start_mask = time.clock()
            mask = get_neck_blur_mask(img_path=origin_img, net=neckMaskNet, dilate=5)
            time_mask = (time.clock() - start_mask)

            debug = False
            start_warp = time.clock()
            if debug:
                warpped_edited_img, debug_img = warp_img(origin_img, edited_img, net=neckMaskNet, debug=True)
            else:
                warpped_edited_img = warp_img(origin_img, edited_img, net=neckMaskNet, debug=False)

            time_warp = (time.clock() - start_warp)

            # cv2.imshow('i', ( warpped_edited_img * (mask / 255)*0.5  + origin_img * (1 - mask / 255)).astype(np.uint8))
            # cv2.waitKey(0)

            res = warpped_edited_img * (mask / 255) + origin_img * (1 - mask / 255)

            res2 = edited_img * (mask / 255) + origin_img * (1 - mask / 255)

            if debug:
                cv2.imwrite(os.path.join(res_dir, f'{image_name}.jpg'),res)
                cv2.imwrite(os.path.join(temp_dir, f'{image_name}_0.jpg'),
                            np.concatenate([origin_img, origin_encode, edited_img,warpped_edited_img, res2, res], axis=1))
                cv2.imwrite(os.path.join(temp_dir, f'{image_name}_1.jpg'),
                            debug_img)
            else:
                cv2.imwrite(os.path.join(res_dir, f'{image_name}.jpg'), res)
                cv2.imwrite(os.path.join(temp_dir, f'{image_name}.jpg'),
                            np.concatenate([origin_img,  res], axis=0))
            times.append([time_edit, time_mask, time_warp])
            img_count += 1
        except:
            pass

    print(times)
    times = np.array(times)
    print(times.shape)


if __name__ == '__main__':
    run()