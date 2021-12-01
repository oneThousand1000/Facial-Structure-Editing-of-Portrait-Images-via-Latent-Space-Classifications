# python 3.6
"""diffuses target images to context images with In-domain GAN Inversion.

Basically, this script first copies the central region from the target image to
the context image, and then performs in-domain GAN inversion on the stitched
image. Different from `intert.py`, masked reconstruction loss is used in the
optimization stage.

NOTE: This script will diffuse every image from `target_image_list` to every
image from `context_image_list`.
"""

import os
import argparse
import numpy as np
import cv2
from CHINGER_inverter import StyleGAN2Inverter
from interface.utils.visualizer import load_image, resize_image, save_image
import glob


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, required=True,
                        help='Directory to save the results. If not specified, '
                             '`data/double_chin_pair/images` will be used by default.')
    parser.add_argument('--latent_space_type', type=str, default='WP',
                        help='latetn_space_type. If not specified, '
                             ' `Wp` will be used by default.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization. (default: 0.01)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations. (default: 100)')

    parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                        help='The perceptual loss scale for optimization. '
                             '(default: 5e-5)')

    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')
    parser.add_argument('-p', '--truncation_psi', type=float, default='0.8')
    return parser.parse_args()





def diffuse(args, img_path, mask_path, latent_path,inverter=None):
    kwargs = {'latent_space_type': args.latent_space_type}
    assert inverter is not None

    image_size = inverter.G.resolution

    image = resize_image(load_image(img_path), (image_size, image_size))
    mask = resize_image(load_image(mask_path), (image_size, image_size))

    image_name = os.path.splitext(os.path.basename(img_path))[0] if isinstance(img_path, str) else 'test.png'

    mask = (mask > 0).astype(np.uint8) * 255
    mask_dilate = cv2.dilate(mask, kernel=np.ones((15,15), np.uint8))
    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(25, 25))
    mask_dilate_blur = mask + (255 - mask) // 255 * mask_dilate_blur

    init_code = np.load(latent_path)


    image_save_path = os.path.join(os.path.join(args.data_dir, 'mask_blur'), '%s.png' % image_name)
    save_image(image_save_path, mask_dilate_blur)

    # Initialize visualizer.
    target_image = image
    code, viz_result = inverter.easy_mask_diffuse(target=target_image,
                                                   init_code=init_code,
                                                   mask=mask_dilate_blur,
                                                   **kwargs)
    latent_code = code
    # Save results.
    assert init_code.shape==code.shape
    latent_code_save_path = os.path.join(os.path.join(args.data_dir, 'codes'),
                                         '%s_inverted_%s_codes.npy' % (image_name,args.latent_space_type))
    np.save(latent_code_save_path, latent_code)


    image_save_path = os.path.join(os.path.join(args.data_dir, 'diffused'), '%s.jpg' % image_name)
    save_image(image_save_path, viz_result)

    image_masked_save_path = os.path.join(os.path.join(args.data_dir, 'res'), '%s.jpg' % image_name)
    image_optimized_save_path = os.path.join(os.path.join(args.data_dir, 'viz'), '%s_op.jpg' % image_name)
    res = viz_result
    #
    origin_img = resize_image(
        load_image(os.path.join(os.path.join(args.data_dir, 'images'), image_name + '_w_doublechin.jpg')),
        (image_size, image_size))
    #
    #
    save_image(image_optimized_save_path, res)
    res = origin_img * (1 - mask_dilate_blur / 255) + res * (mask_dilate_blur / 255)
    save_image(image_masked_save_path, res)
    #
    res_path = os.path.join(os.path.join(args.data_dir, 'viz'), '%s.jpg' % image_name)

    save_img =np.concatenate([origin_img,res],axis=1)


    save_image(res_path, save_img)


def run():
    args = parse_args()
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.exists(os.path.join(args.data_dir, 'diffused')):
        os.mkdir(os.path.join(args.data_dir, 'diffused'))
    if not os.path.exists(os.path.join(args.data_dir, 'codes')):
        os.mkdir(os.path.join(args.data_dir, 'codes'))
    if not os.path.exists(os.path.join(args.data_dir, 'mask_blur')):
        os.mkdir(os.path.join(args.data_dir, 'mask_blur'))
    if not os.path.exists(os.path.join(args.data_dir, 'res')):
        os.mkdir(os.path.join(args.data_dir, 'res'))
    if not os.path.exists(os.path.join(args.data_dir, 'viz')):
        os.mkdir(os.path.join(args.data_dir, 'viz'))


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    image_list = glob.glob(os.path.join(args.data_dir, 'simple_synthesis/*.jpg'))
    mask_list = [path.replace('simple_synthesis', 'mask').replace('jpg', 'png') for path in
                 image_list]
    latent_list = [path.replace('simple_synthesis', 'codes').replace('.jpg', '_wp.npy') for path in
                 image_list]
    image_num = len(image_list)

    model_name = 'stylegan2_ffhq'
    assert args.latent_space_type in ['W','w','WP','wp']
    inverter = StyleGAN2Inverter(
        model_name,
        learning_rate=args.learning_rate,
        iteration=args.num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=args.loss_weight_feat,
        truncation_psi=args.truncation_psi,
        logger=None)


    for index in range(image_num):

        image_name = os.path.splitext(os.path.basename(image_list[index]))[0]
        print('diffuse %s' % (os.path.join(os.path.join(args.data_dir, 'res'), image_name+'.jpg')))
        if (not os.path.exists(
                os.path.join(os.path.join(args.data_dir, 'diffused'), image_name+'.jpg'))):
            diffuse(args=args, img_path=image_list[index], mask_path=mask_list[index],latent_path=latent_list[index], inverter=inverter)


if __name__ == '__main__':
    run()


















