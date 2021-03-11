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
from interface.utils.manipulator import train_boundary,project_boundary

# [2020-12-28 19:42:55,342][INFO] Spliting training and validation sets:
# [2020-12-28 19:42:55,394][INFO]   Training: 6770 positive, 6770 negative.
# [2020-12-28 19:42:55,396][INFO]   Validation: 753 positive, 753 negative.
# [2020-12-28 19:42:55,396][INFO]   Remaining: 243 positive, 38453 negative.
# [2020-12-28 19:42:55,396][INFO] Training boundary.


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Train semantic boundary with given latent codes and '
                    'attribute scores.')
    parser.add_argument('-o', '--output_dir', type=str,default='boundaries/fine/double_chin_wp_all',
                        help='Directory to save the output results. (required)')
    parser.add_argument('-c', '--latent_codes_path', type=str, default='./data/wps_all.npy',
                        help='Path to the input latent codes. (required)')
    parser.add_argument('-s', '--scores_path', type=str, default='./data/score_all.npy',
                        help='Path to the input attribute scores. (required)')
    parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.49,
                        help='How many samples to choose for training. '
                             '(default: 0.2)')
    parser.add_argument('-r', '--split_ratio', type=float, default=0.98,
                        help='Ratio with which to split training and validation '
                             'sets. (default: 0.7)')
    parser.add_argument('-V', '--invalid_value', type=float, default=None,
                        help='Sample whose attribute score is equal to this '
                             'field will be ignored. (default: None)')

    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='generate_data')
    latent_codes_all = np.load(args.latent_codes_path)

    print(latent_codes_all.shape)
    scores = np.load(args.scores_path)

    scores=np.reshape(scores,(scores.shape[0],1))
    print("scores dim:", scores.shape)
    boundarys=[]
    for i in range(18):

        latent_codes = np.reshape(latent_codes_all[:,i,:], (latent_codes_all.shape[0], 512))


        print("latent_space_dim:", latent_codes.shape)
        boundary = train_boundary(latent_codes=latent_codes,
                                  scores=scores,
                                  chosen_num_or_ratio=args.chosen_num_or_ratio,
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
