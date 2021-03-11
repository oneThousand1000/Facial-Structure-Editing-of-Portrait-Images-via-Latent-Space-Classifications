# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semantics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""

import os.path
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm
import glob
from styleGAN2_model.model_settings import MODEL_POOL
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from  classifier.src.feature_extractor.neck_mask_extractor import get_neck_mask,get_parsingNet
import random

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Generate images with given model.')
    parser.add_argument('-m', '--model_name', type=str, default='stylegan2_ffhq',
                        choices=list(MODEL_POOL),
                        help='Name of the model for generation. (required)')
    parser.add_argument('-o', '--output_dir', type=str, default='data/test',
                        help='Directory to save the output results. (required)')
    parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                        help='If specified, will load latent codes from given '
                             'path instead of randomly sampling. (optional)')
    parser.add_argument('-n', '--num', type=int, default=10,
                        help='Number of images to generate. This field will be '
                             'ignored if `latent_codes_path` is specified. '
                             '(default: 1)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('-S', '--generate_style', action='store_true',
                        help='If specified, will generate layer-wise style codes '
                             'in Style GAN. (default: do not generate styles)')
    parser.add_argument('-I', '--generate_image', action='store_false',
                        help='If specified, will skip generating images in '
                             'Style GAN. (default: generate images)')

    parser.add_argument('-O', '--double_chin_only', action='store_true',
                        help='If specified, only generate double chin img. (default: generate images)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()




    model = StyleGAN2Generator(args.model_name, None,truncation_psi=0.8)

    print('label size:',model.model.label_size)
    latent_space_type='w'
    kwargs = {'latent_space_type': latent_space_type}

    #boundary=np.load('./boundaries/double_chin_wp_psi_0.8/boundary.npy')
    wp_path='./samples/000000_w.npy'
    latent_code = np.load(wp_path)



    latent_code = np.reshape(latent_code, (1, 512))
    print(latent_code.shape)
    outputs = model.easy_synthesize(latent_code,
                                    **kwargs)
    after = outputs['image'][0][:, :, ::-1]
    cv2.imwrite('./000000_w.jpg', after)
    outputs2 = model.easy_synthesize(latent_code,
                                    **kwargs)
    after2 = outputs2['image'][0][:, :, ::-1]
    cv2.imwrite('./000000_w2.jpg', after2)
#----------------
    latent_space_type = 'wp'
    kwargs = {'latent_space_type': latent_space_type}

    # boundary=np.load('./boundaries/double_chin_wp_psi_0.8/boundary.npy')
    wp_path = './samples/000000_wp.npy'
    latent_code = np.load(wp_path)

    # latent_code[:, range(7, 18), :] *= 1 - 0.8
    # latent_code[:, range(7, 18), :] += latent_code1[:, range(7, 18), :] * 0.8

    latent_code = np.reshape(latent_code, (1,18, 512))
    print(latent_code.shape)
    outputs = model.easy_synthesize(latent_code,
                                    **kwargs)
    after = outputs['image'][0][:, :, ::-1]
    cv2.imwrite('./000000_wp.jpg', after)
    outputs2 = model.easy_synthesize(latent_code,
                                     **kwargs)
    after2 = outputs2['image'][0][:, :, ::-1]
    cv2.imwrite('./000000_wp2.jpg', after2)







if __name__ == '__main__':
    main()

