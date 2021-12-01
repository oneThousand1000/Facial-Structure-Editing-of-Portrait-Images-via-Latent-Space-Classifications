# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np
import glob
from interface.utils.logger import setup_logger
from interface.utils.manipulator import train_boundary


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Train semantic boundary with given latent codes and '
                    'attribute scores.')
    parser.add_argument('-o', '--output_dir', type=str,required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                        help='Path to the input latent codes. (required)')

    parser.add_argument('-r', '--split_ratio', type=float, default=0.9,
                        help='Ratio with which to split training and validation '
                             'sets. (default: 0.7)')
    parser.add_argument('-V', '--invalid_value', type=float, default=None,
                        help='Sample whose attribute score is equal to this '
                             'field will be ignored. (default: None)')

    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='generate_data')


    latent_codes_all = []
    scores=[]
    for code_path in glob.glob(os.path.join(args.latent_codes_path,'*_inverted_WP_codes.npy')):
        name= os.path.basename(code_path)[:6]
        code_path_origin=os.path.join(args.latent_codes_path,f'{name}_wp.npy')
        if os.path.exists(code_path_origin):
            latent_codes_all.append(np.load(code_path))
            latent_codes_all.append(np.load(code_path_origin))
            scores.append(0)
            scores.append(1)

    latent_codes_all=np.concatenate(latent_codes_all,axis=0)
    scores = np.array(scores)
    scores=np.reshape(scores,(scores.shape[0],1))
    print("scores dim:", scores.shape)
    boundarys=[]
    assert latent_codes_all.shape[1]==18
    for i in range(18):

        latent_codes = np.reshape(latent_codes_all[:,i,:], (latent_codes_all.shape[0], 512))


        print("latent_space_dim:", latent_codes.shape)
        boundary = train_boundary(latent_codes=latent_codes,
                                  scores=scores,
                                  chosen_num_or_ratio=0.5,
                                  split_ratio=args.split_ratio,
                                  invalid_value=args.invalid_value,
                                  logger=logger)
        print(boundary.shape)
        np.save(os.path.join(args.output_dir, f'boundary_{i}.npy'), boundary)

        boundarys.append(boundary)

    boundarys = np.concatenate(boundarys, axis=1)
    boundarys = np.reshape(boundarys, (1, 18, 512))

    np.save(os.path.join(args.output_dir, f'boundary.npy'), boundarys)



if __name__ == '__main__':
    main()
