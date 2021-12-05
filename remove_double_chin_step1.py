
import os.path
import argparse
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from  classifier.src.feature_extractor.neck_mask_extractor import get_neck_mask,get_parsingNet
def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-b', '--boundary_path', type=str, required=True,
                        help='Path to the semantic boundary. (required)')
    parser.add_argument('-i', '--input_data_dir', type=str,required=True,
                        help='load latent codes ')
    parser.add_argument('--alpha', type=float, default=-4.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('-p', '--truncation_psi', type=float, default='0.8')
    return parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def run():
    model_name='stylegan2_ffhq'

    args = parse_args()
    latent_space_type = args.latent_space_type
    #logger = setup_logger(args.output_dir, logger_name='generate_data')
    steps = 2
    mkdir(args.output_dir)
    mkdir(os.path.join(args.output_dir, 'images'))
    mkdir(os.path.join(args.output_dir, 'codes'))
    mkdir(os.path.join(args.output_dir, 'mask'))
    mkdir(os.path.join(args.output_dir, 'simple_synthesis'))


    # -------------------------interpolate image-----------------------------

    print(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger=None,truncation_psi=args.truncation_psi)
    kwargs = {'latent_space_type': latent_space_type}

    print(f'Preparing boundary.')
    boundary_path = os.path.join(args.boundary_path, 'boundary.npy')
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{boundary_path}` does not exist!')
    boundary = np.load(boundary_path)

    np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)
    print(f'Preparing latent codes.')

    wp_latent_path = os.path.join(args.input_data_dir, 'wp.npy')
    if not os.path.isfile(wp_latent_path):
        raise ValueError(f'Boundary `{wp_latent_path}` does not exist!')
    latent_codes = np.load(wp_latent_path)

    double_chin_scores_path = os.path.join(args.input_data_dir, 'double_chin_scores.npy')
    if not os.path.isfile(boundary_path):
        raise ValueError(f'Boundary `{double_chin_scores_path}` does not exist!')
    double_chin_scores = np.load(double_chin_scores_path)

    latent_codes = latent_codes[double_chin_scores[:,0],:,:]

    np.save(os.path.join(args.output_dir, f'latent_codes_{latent_space_type}.npy'), latent_codes)

    total_num = latent_codes.shape[0]

    print(f'Editing {total_num} samples.')

    results = []

    #net=get_net()
    parsingNet=get_parsingNet()
    image_name = ['w_doublechin', 'wo_doublechin']
    for sample_id in tqdm(range(total_num), leave=False):
        latent=latent_codes[sample_id:sample_id + 1]
        interpolations=np.concatenate([latent,latent+boundary*args.alpha],axis=0)

        interpolation_id = 0
        image_pair=[]

        style_latent=None
        origin_img=None
        origin_mask=None

        for interpolations_batch in model.get_batch_inputs(interpolations):

            if interpolation_id==0:
                outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                style_latent=interpolations_batch
            else:
                assert style_latent.any()!=None
                outputs = model.easy_style_mixing(latent_codes=interpolations_batch,
                                                  style_range=range(7,18),
                                                  style_codes=style_latent,
                                                  mix_ratio=0.8,
                                                             **kwargs
                                                             )
            info=defaultdict(list)
            if interpolation_id==0:
                for key, val in outputs.items():
                    if key == 'w' or key=='wp':
                        save_path = os.path.join(os.path.join(args.output_dir, 'codes'),
                                                 f'{sample_id:06d}_{key}.npy')
                        np.save(save_path, val)


            assert len(outputs['image'])==1
            for image in outputs['image']:
                save_path = os.path.join(os.path.join(args.output_dir,'images'),
                                         f'{sample_id:06d}_{image_name[interpolation_id]}.jpg')
                cv2.imwrite(save_path, image[:, :, ::-1])

                info['path']=save_path

                if interpolation_id == 0:
                    neck_mask=get_neck_mask(img_path=save_path,net=parsingNet)



                    mask_path = os.path.join(os.path.join(args.output_dir, 'mask'),
                                             f'{sample_id:06d}.png')
                    cv2.imwrite(mask_path, neck_mask)

                    origin_img = image[:, :, ::-1]
                    origin_mask = neck_mask


                else:
                    synthesis_image = origin_img * (1 - origin_mask // 255) + \
                                      image[:, :, ::-1] * (origin_mask // 255)
                    save_path = os.path.join(os.path.join(args.output_dir, 'simple_synthesis'),
                                             f'{sample_id:06d}.jpg')

                    cv2.imwrite(save_path, synthesis_image)



            interpolation_id += 1
            image_pair.append(info)
        results.append(image_pair)
        assert interpolation_id == steps
    print(f'Successfully edited {total_num} samples.')

    print(f'Initializing parsingNet.')




if __name__ == '__main__':
    run()




