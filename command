python remove_double_chin_step1.py --output_dir F:/DoubleChin/datasets/ffhq_data/psi_0.8_fixed_gen  --boundary_path ./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w/boundary.npy   --input_latent_codes_path  ./data/double_chin_wp_0.8.npy
python remove_double_chin_step2.py --data_dir F:/DoubleChin/datasets/ffhq_data/test

python generate_data_and_score.py --output_dir F:/DoubleChin/datasets/ffhq_data/psi_0.8_double_chin_S_space