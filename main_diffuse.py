import os.path
import argparse
import cv2
import numpy as np
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from classifier.src.feature_extractor.neck_mask_extractor import get_neck_mask, get_parsingNet
from classifier.classify import get_model, check_double_chin

#from interface.utils.myinverter import StyleGAN2Inverter
from CHINGER_inverter import StyleGAN2Inverter
import glob
import time

from utils import str2bool
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
    parser.add_argument('-i', '--data_dir', type=str, required=True,
                        help='If specified, will load latent codes from given ')

    parser.add_argument('-b', '--boundary_path', type=str,
                        required=True,
                        help='Path to the semantic boundary. (required)')

    parser.add_argument('--boundary_init_ratio', type=float, default=-4.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('--boundary_additional_ratio', type=float, default=-1.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization. (default: 0.01)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations. (default: 100)')

    parser.add_argument('--loss_weight_feat', type=float, default=1e-4,
                        help='The perceptual loss scale for optimization. '
                             '(default: 5e-5)')

    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')

    parser.add_argument("--cycle", type=str2bool, nargs='?',
                        const=False, default=False,
                        help="diffuse until no double chin.")

    return parser.parse_args()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run():
    model_name = 'stylegan2_ffhq'
    args = parse_args()
    latent_space_type = args.latent_space_type

    assert os.path.exists(args.data_dir), f'data_dir {args.data_dir} dose not exist!'
    origin_img_dir = os.path.join(args.data_dir, 'origin')
    code_dir = os.path.join(args.data_dir, 'code')
    diffuse_code_dir = os.path.join(args.data_dir, 'diffuse_code')
    res_dir = os.path.join(args.data_dir, 'diffuse_res')
    assert os.path.exists(origin_img_dir), f'{origin_img_dir} dose not exist!'
    assert os.path.exists(code_dir), f'data_dir {code_dir} dose not exist!'
    mkdir(res_dir)
    mkdir(diffuse_code_dir)

    print(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger=None)
    kwargs = {'latent_space_type': latent_space_type}

    print(f'Initializing Inverter.')
    inverter = StyleGAN2Inverter(
        model_name,
        learning_rate=args.learning_rate,
        iteration=args.num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=args.loss_weight_feat,
        logger=None,
        stylegan2_model=model)

    print(f'Preparing boundary.')
    boundary_path = os.path.join(args.boundary_path, 'boundary.npy')
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{boundary_path}` does not exist!')
    boundary = np.load(boundary_path)

    print(f'Load latent codes and images from `{args.data_dir}`.')
    latent_codes = []
    origin_img_list = []
    for img in glob.glob(os.path.join(origin_img_dir, '*'))[::-1]:
        name = os.path.basename(img)[:-4]
        code_path = os.path.join(code_dir, f'{name}_wp.npy')
        if os.path.exists(code_path):
            latent_codes.append(code_path)
            origin_img_list.append(img)
    total_num = len(latent_codes)

    print(f'Processing {total_num} samples.')

    neckMaskNet = get_parsingNet()
    double_chin_checker = get_model()

    times = []
    for img_index in range(total_num):
        score = 1
        image_name = os.path.splitext(os.path.basename(origin_img_list[img_index]))[0]

        if os.path.exists(os.path.join(code_dir, f'{image_name}_inverted_wp.npy')):
            continue

        wps_latent = np.reshape(np.load(latent_codes[img_index]), (1, 18, 512))
        origin_img = cv2.imread(origin_img_list[img_index])

        neck_mask = get_neck_mask(img_path=origin_img, net=neckMaskNet)

        neck_mask = (neck_mask > 0).astype(np.uint8) * 255
        mask_dilate = cv2.dilate(neck_mask, kernel=np.ones((30, 30), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(35, 35))
        mask_dilate_blur = neck_mask + (255 - neck_mask) // 255 * mask_dilate_blur
        train_count = 0
        ratio = args.boundary_init_ratio
        while (score):
            train_count += 1
            edited_wps_latent = wps_latent + ratio * boundary

            edited_output = model.easy_style_mixing(latent_codes=edited_wps_latent,
                                                    style_range=range(7, 18),
                                                    style_codes=wps_latent,
                                                    mix_ratio=1.0, **kwargs)
            edited_img = edited_output['image'][0][:, :, ::-1]

            synthesis_image = origin_img * (1 - neck_mask // 255) + \
                              edited_img * (neck_mask // 255)
            init_code = wps_latent

            target_image = synthesis_image[:, :, ::-1]

            start_diffuse = time.clock()
            code, viz_result = inverter.easy_mask_diffuse(target=target_image,
                                                          init_code=init_code,
                                                          mask=mask_dilate_blur,
                                                          **kwargs)

            time_diffuse = (time.clock() - start_diffuse)

            times.append(time_diffuse)
            viz_result = viz_result[:, :, ::-1]
            res = origin_img * (1 - mask_dilate_blur / 255) + viz_result * (mask_dilate_blur / 255)
            score = check_double_chin(img=res, model=double_chin_checker)
            if score:
                print('\n still exists double chin! continue....')
            else:
                print('\n double chin is removed')

            wps_latent = code
            ratio += args.boundary_additional_ratio

            if not args.cycle or train_count >= 5:
                break


        print('train %d times.' % train_count)
        np.save(os.path.join(diffuse_code_dir, f'{image_name}_inverted_wp.npy'), code)
        cv2.imwrite(os.path.join(res_dir, f'{image_name}.jpg'), res)


if __name__ == '__main__':
    run()
