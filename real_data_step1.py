
import os.path
import argparse
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from interface.utils.manipulator import linear_interpolate
from classifier.src.feature_extractor.neck_mask_extractor import get_neck_mask,get_parsingNet
import glob
def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')
    parser.add_argument('-i', '--data_dir', type=str, default='F:/DoubleChin/datasets/ffhq_data/test',
                        help='If specified, will load latent codes from given ')
    
    parser.add_argument('-b', '--boundary_path', type=str, default='./interface/boundaries/psi_0.8/stylegan2_ffhq_double_chin_w/boundary.npy',
                        help='Path to the semantic boundary. (required)')

    parser.add_argument('--start_distance', type=float, default=0.0,
                        help='Start point for manipulation in latent space. '
                             '(default: -3.0)')
    parser.add_argument('--end_distance', type=float, default=-3.0,
                        help='End point for manipulation in latent space. '
                             '(default: 3.0)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='wp',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')

    return parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def run():
    model_name='stylegan2_ffhq'

    args = parse_args()
    latent_space_type = args.latent_space_type
    #logger = setup_logger(args.data_dir, logger_name='generate_data')
    steps = 2
    mkdir(args.data_dir)
    mkdir(os.path.join(args.data_dir, 'images'))
    #mkdir(os.path.join(args.data_dir, 'codes'))
    mkdir(os.path.join(args.data_dir, 'mask'))
    mkdir(os.path.join(args.data_dir, 'simple_synthesis'))
    mkdir(os.path.join(args.data_dir, 'origin'))

    # -------------------------interpolate image-----------------------------

    print(f'Initializing generator.')
    model = StyleGAN2Generator(model_name, logger=None)
    kwargs = {'latent_space_type': latent_space_type}

    print(f'Preparing boundary.')
    if not os.path.isfile(args.boundary_path):
        raise ValueError(f'Boundary `{args.boundary_path}` does not exist!')
    boundary = np.load(args.boundary_path)

    np.save(os.path.join(args.data_dir, 'boundary.npy'), boundary)
    print(f'Preparing latent codes.')


    print(f'Load latent codes and images from `{args.data_dir}`.')
    latent_codes=[]
    origin_img_list = []
    for img in glob.glob( os.path.join(os.path.join(args.data_dir, 'origin'),'*.jpg')):
        name=os.path.basename(img)[:6]
        code_path=os.path.join(os.path.join(args.data_dir, 'code'),
                                                  f'{name}_wp.npy')
        #print(img,code_path)
        #print(code_path)
        if os.path.exists(code_path):
            latent_codes.append(code_path)
            origin_img_list.append(img)




    print(len(latent_codes),len(origin_img_list))
   #  np.save(os.path.join(args.data_dir, f'latent_codes_{latent_space_type}.npy'), latent_codes)

    total_num = len(latent_codes)

    print(f'Editing {total_num} samples.')

    results = []

    #net=get_net()
    parsingNet=get_parsingNet()
    for index in tqdm(range(total_num), leave=False):

        #print(latent_codes[index],origin_img_list[index])
        sample_id=int(os.path.basename(origin_img_list[index])[:-4])
        zs_latent=np.reshape(np.load(latent_codes[index]),(1,18,512))
        #print(origin_img_list[index],latent_codes[index])
        origin_img = cv2.imread(origin_img_list[index])
        latent=zs_latent
        interpolations = linear_interpolate(latent,
                                            boundary,
                                            start_distance=args.start_distance,
                                            end_distance=args.end_distance,
                                            steps=steps)
        image_name=['w_doublechin','wo_doublechin']
        interpolation_id = 0
        image_pair=[]

        style_latent=None
        origin_mask=None
        if os.path.exists( os.path.join(os.path.join(args.data_dir,'images'),
                                         f'{sample_id:06d}_{image_name[interpolation_id]}.jpg')):
            continue
        for interpolations_batch in model.get_batch_inputs(interpolations):
            if interpolation_id==0:
                outputs = model.easy_synthesize(interpolations_batch, **kwargs)
                style_latent=interpolations_batch
            else:
                assert style_latent.any()!=None
                outputs = model.easy_style_mixing(latent_codes=interpolations_batch,
                                                  style_range=range(7,18),
                                                  style_codes=style_latent,
                                                  mix_ratio=1.0,
                                                             **kwargs
                                                             )
            info=defaultdict(list)
            # if interpolation_id==0:
            #     for key, val in outputs.items():
            #         if key == 'w' or key=='wp':
            #             save_path = os.path.join(os.path.join(args.data_dir, 'codes'),
            #                                      f'{sample_id:06d}_{key}.npy')
            #             np.save(save_path, val)


            assert len(outputs['image'])==1
            for image in outputs['image']:
                save_path = os.path.join(os.path.join(args.data_dir,'images'),
                                         f'{sample_id:06d}_{image_name[interpolation_id]}.jpg')
                cv2.imwrite(save_path, image[:, :, ::-1])

                info['path']=save_path

                if interpolation_id == 0:

                    #neck_mask = neck_mask_gen(img=save_path, net=net)
                    neck_mask = get_neck_mask(img_path=save_path, net=parsingNet)

                    mask_path = os.path.join(os.path.join(args.data_dir, 'mask'),
                                             f'{sample_id:06d}.png')
                    cv2.imwrite(mask_path, neck_mask)


                    origin_mask = neck_mask


                else:
                    # origin_img =
                    synthesis_image = origin_img * (1 - origin_mask // 255) + \
                                      image[:, :, ::-1] * (origin_mask // 255)
                    save_path = os.path.join(os.path.join(args.data_dir, 'simple_synthesis'),
                                             f'{sample_id:06d}.jpg')

                    cv2.imwrite(save_path, synthesis_image)

                # full_mask, full_mask_wo_hair = get_full_mask(img_path=save_path, net=parsingNet)
                # mask_path = os.path.join(os.path.join(args.data_dir, 'full_mask'),
                #                          f'{sample_id:06d}_{image_name[interpolation_id]}.png')
                #
                # cv2.imwrite(mask_path, full_mask)
                # mask_path = os.path.join(os.path.join(args.data_dir, 'full_mask_wo_hair'),
                #                          f'{sample_id:06d}_{image_name[interpolation_id]}.png')
                #
                # cv2.imwrite(mask_path, full_mask_wo_hair)



            interpolation_id += 1
            image_pair.append(info)
        results.append(image_pair)
        assert interpolation_id == steps
    print(f'Successfully edited {total_num} samples.')

    print(f'Initializing parsingNet.')




if __name__ == '__main__':
    run()




