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
    parser.add_argument('-o', '--output_dir', type=str,default='boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_z',
                        help='Directory to save the output results. (required)')
    parser.add_argument('-c', '--latent_codes_path', type=str, default='F:/DoubleChin/datasets/ffhq_gen_data/stylegan_ffhq_2_psi_0.8/z.npy',
                        help='Path to the input latent codes. (required)')
    parser.add_argument('-s', '--scores_path', type=str, default='F:/DoubleChin/datasets/ffhq_gen_data/stylegan_ffhq_2_psi_0.8/double_chin_scores.npy',
                        help='Path to the input attribute scores. (required)')
    parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.1,
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
    """Main function."""
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='train_boundary')

    logger.info('Loading latent codes.')
    if not os.path.isfile(args.latent_codes_path):
        raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
    latent_codes = np.load(args.latent_codes_path)

    logger.info('Loading attribute scores.')

    if not os.path.isfile(args.scores_path):
        raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
    scores = np.load(args.scores_path)
    print("scores dim:", scores.shape)
    latent_codes=np.reshape(latent_codes,(latent_codes.shape[0],-1))

    latent_codes=latent_codes[:scores.shape[0],:]
    print("latent_space_dim:", latent_codes.shape)
    boundary = train_boundary(latent_codes=latent_codes,
                              scores=scores,
                              chosen_num_or_ratio=args.chosen_num_or_ratio,
                              split_ratio=args.split_ratio,
                              invalid_value=args.invalid_value,
                              logger=logger)
    print(boundary.shape)
    np.save(os.path.join(args.output_dir,'boundary.npy'),boundary)




if __name__ == '__main__':
    main()
