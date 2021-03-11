
import os.path
import argparse
import numpy as np

from interface.utils.logger import setup_logger
from interface.utils.manipulator import train_boundary,project_boundary











double_chin_b = np.load('F:/DoubleChin/styleGAN_related/interfacegan/boundaries/stylegan_ffhq_age_boundary.npy')
boundary=np.load('F:/DoubleChin/styleGAN_related/interfacegan/boundaries/stylegan_ffhq_eyeglasses_boundary.npy')
res = np.load('F:/DoubleChin/styleGAN_related/interfacegan/boundaries/stylegan_ffhq_eyeglasses_c_age_boundary.npy')


#conditional_double_chin_b =
print(res-project_boundary(double_chin_b, boundary))
print(res-project_boundary(boundary, double_chin_b))


# os.mkdir('boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w_c_pose/')
# np.save(os.path.join('boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w_c_pose/', 'boundary.npy'), conditional_double_chin_b)