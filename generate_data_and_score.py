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

from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from interface.utils.logger import setup_logger
from classifier.classify import get_model,check_double_chin
from utils import str2bool

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Generate images with given model.')

    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                        help='If specified, will load latent codes from given '
                             'path instead of randomly sampling. (optional)')
    parser.add_argument('-n', '--num', type=int, default=50000,
                        help='Number of images to generate. This field will be '
                             'ignored if `latent_codes_path` is specified. '
                             '(default: 1)')
    parser.add_argument('-S', '--generate_style', action='store_true',
                        help='If specified, will generate layer-wise style codes '
                             'in Style GAN. (default: do not generate styles)')
    parser.add_argument('-I', '--generate_image', action='store_false',
                        help='If specified, will skip generating images in '
                             'Style GAN. (default: generate images)')
    parser.add_argument('-p', '--truncation_psi', type=float,default='0.8')
    parser.add_argument("--double_chin_only", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Only generate double chin images.")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    model_name='stylegan2_ffhq'
    logger = setup_logger(args.output_dir, logger_name='generate_data')

    double_chin_checker=get_model()

    logger.info(f'Initializing generator.')

    model = StyleGAN2Generator(model_name, logger,truncation_psi=args.truncation_psi)

    kwargs = {'latent_space_type':'z'}

    logger.info(f'Preparing latent codes.')

    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)

    total_num = latent_codes.shape[0]

    logger.info(f'Generating {total_num} samples.')
    results = defaultdict(list)
    pbar = tqdm(total=total_num, leave=False)
    scores = []
    count_double_chin = 0
    count=0
    if args.double_chin_only:
        print('Only generate images that have double chin!')


    for latent_codes_batch in model.get_batch_inputs(latent_codes):
        count+=1
        outputs = model.easy_synthesize(latent_codes_batch,
                                        **kwargs,
                                        generate_style=args.generate_style,
                                        generate_image=args.generate_image)
        if args.double_chin_only:
            choose = []
            key='image'
            val=outputs[key]
            for image in val:


                score = check_double_chin(img=image[:, :, ::-1], model=double_chin_checker)
                pbar.update(1)
                if (score == 1):
                    choose.append(True)
                    scores.append(score)
                    save_path = os.path.join(args.output_dir, f'{count_double_chin + 50186:06d}.jpg')
                    cv2.imwrite(save_path, image[:, :, ::-1])
                    count_double_chin += 1
                else:
                    choose.append(False)
                    #os.remove(save_path)
            for key, val in outputs.items():
                if  not key == 'image':
                    if choose[0]:
                        results[key].append(val)
        else:
            for key, val in outputs.items():
                if key == 'image':
                    for image in val:
                        save_path = os.path.join(args.output_dir, f'{pbar.n:06d}.jpg')
                        cv2.imwrite(save_path, image[:, :, ::-1])

                        score = check_double_chin(img=save_path, model=double_chin_checker)
                        scores.append(score)
                        if (score == 1):
                            count_double_chin += 1

                        pbar.update(1)
                else:
                    results[key].append(val)


        if 'image' not in outputs:
            pbar.update(latent_codes_batch.shape[0])
        if pbar.n % 1000 == 0 or pbar.n == total_num:
            logger.debug(f'  Finish {pbar.n:6d} samples.')
    pbar.close()

    logger.info(f'Saving results.')
    for key, val in results.items():
        save_path = os.path.join(args.output_dir, f'{key}.npy')
        np.save(save_path, np.concatenate(val, axis=0))
        print( np.concatenate(val, axis=0).shape)

    score_save_path=os.path.join(args.output_dir,'double_chin_scores.npy')
    scores_array=np.array(scores)[:,np.newaxis]
    print(scores_array.shape)
    np.save(score_save_path,scores_array)
    print('%d double chin images'%count_double_chin)

    double_chin_w=[]
    for i in range(len(results['w'])):
        if scores_array[i]==1:
            double_chin_w.append(results['w'][i])

    save_path = os.path.join(args.output_dir, 'double_chin_w.npy')
    np.save(save_path, np.concatenate(double_chin_w, axis=0))

if __name__ == '__main__':
    main()
