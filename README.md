# Coarse-to-Fine: Facial Structure Editing of Portrait Images via Latent Space Classifications

![teaser](./docs/teaser.jpg)

Published in *ACM Transactions on Graphics (Proc. of Siggraph 2021), 40(4): Article 46.*, 2021

[Yiqian Wu](https://onethousandwu.com/), [Yongliang Yang](https://www.yongliangyang.net/), Qinjie Xiao, [Xiaogang Jin](http://www.cad.zju.edu.cn/home/jin/).

<div align="center">

[![Project](https://img.shields.io/badge/Doublechin%20Removal-1?label=Project&color=8B93FF)](https://onethousandwu.com/doublechinremoval.github.io/)
[![Paper](https://img.shields.io/badge/Main%20Paper-1?color=58A399)](https://dl.acm.org/doi/10.1145/3450626.3459814)
[![Suppl](https://img.shields.io/badge/Supplementary-1?color=378CE7)](https://drive.google.com/file/d/14oIdiv2NkvpRYxomDRq0AQEpBuL4pKtv/view?usp=sharing)
[![Video](https://img.shields.io/badge/Video-1?color=FDA403)](https://youtu.be/1aYPceNkwIQ)
[![Dataset](https://img.shields.io/badge/Dataset-1?color=FC819E)](https://github.com/oneThousand1000/coarse-to-fine-chin-editing)
[![Github](https://img.shields.io/github/stars/oneThousand1000/Facial-Structure-Editing-of-Portrait-Images-via-Latent-Space-Classifications)](https://github.com/oneThousand1000/Facial-Structure-Editing-of-Portrait-Images-via-Latent-Space-Classifications)

</div>


**Abstract:**

Facial structure editing of portrait images is challenging given the facial variety, the lack of ground-truth, the necessity of jointly adjusting color and shape, and the requirement of no visual artifacts. In this paper, we investigate how to perform chin editing as a case study of editing facial structures. We present a novel method that can automatically remove the double chin effect in portrait images. Our core idea is to train a fine classification boundary in the latent space of the portrait images. This can be used to edit the chin appearance by manipulating the latent code of the input portrait image while preserving the original portrait features. To achieve such a fine separation boundary, we employ a carefully designed training stage based on latent codes of paired synthetic images with and without a double chin. In the testing stage, our method can automatically handle portrait images with only a refinement to subtle misalignment before and after double chin editing. Our model enables alteration to the neck region of the input portrait image while keeping other regions unchanged, and guarantees the rationality of neck structure and the consistency of facial characteristics. To the best of our knowledge, this presents the first effort towards an effective application for editing double chins. We validate the efficacy and efficiency of our approach through extensive experiments and user studies.

## License

<u>**You can use, redistribute, and adapt this software for NON-COMMERCIAL purposes only**.</u>

**For business inquiries, please contact onethousand@zju.edu.cn / onethousand1250@gmail.com / jin@cad.zju.edu.cn**

## Requirements

1. Windows 
2. Python 3.6
3. NVIDIA GPU + CUDA10.0 + CuDNN (also tested in  CUDA10.1)

## Installation

1. Download the following pretrained models, put each of them to **PATH**:

| model                                                        | PATH                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [classification_model.pth](https://drive.google.com/file/d/1qztscqTs_6caoSQQ6I9E2h20_SwZhRFo/view?usp=sharing) | ./classifier/double_chin_classification                      |
| [79999_iter.pth](https://drive.google.com/file/d/1eP90uPItdAy1czivugAM3ZK68OdY2pfe/view?usp=sharing) | ./classifier/src/feature_extractor/face_parsing_PyTorch/res/cp |
| [Gs.pth](https://drive.google.com/file/d/1ftka1OI8pvMmml6Qz4Mu4reqH_5h94Mx/view?usp=sharing) | ./styleGAN2_model/pretrain                                   |
| [vgg16.pth](https://drive.google.com/file/d/1V8r8WqDp5vHvE6ooV70h1n7KomF8AbER/view?usp=sharing) | ./styleGAN2_model/pretrain                                   |
| [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) | ./models                                                     |

2. Create conda environment:

```python
conda create -n Coarse2Fine python=3.6
activate Coarse2Fine
```

3. Then install other dependencies by

```
pip install -r requirements.txt
```

## How to Use

### Pre-trained separation boundaries

Pre-trained separation boundaries can be found at [./interface/boundaries](https://github.com/oneThousand1000/CHINGER-Removing-Double-Chin-from-Portraits/tree/main/interface/boundaries):

| dir         | information                                                  |
| ----------- | ------------------------------------------------------------ |
| ├ coarse    | **coarse** separation boundaries of StyleGAN2                |
| │ ├ psi_0.5 | coarse separation boundaries trained from **psi-0.5** dataset |
| │ └ psi_0.8 | coarse separation boundaries trained from **psi-0.8** dataset |
| ├ fine      | **fine** separation boundaries of StyleGAN2                  |
| │ ├ psi_0.5 | fine separation boundaries trained from **psi-0.5** dataset  |
| │ ├ psi_0.8 | fine separation boundaries trained from **psi-0.8** dataset  |
| └  └ all    | fine separation boundaries trained from **overall** dataset  |

Notice that **psi-0.5** dataset and **psi-0.8** dataset are generated by stylegan2 with **psi=0.5(faces are more stable )** and **psi=0.8(faces are more diverse)**

### Testing

#### data prepare :

For<u> **real images**</u>, first find the matching latent vectors.

First, [align faces from input images](https://github.com/pbaylies/stylegan-encoder/blob/master/align_images.py) and save aligned images `{name}.jpg`  to DATA_PATH/origin.

```python
python align_images.py --raw_dir DATA_PATH/raw  --aligned_dir DATA_PATH/origin
```

Second, we recommend to use the projector of official **[ stylegan2 ](https://github.com/NVlabs/stylegan2)** to obtain the latent codes of real images, to correctly run the StyleGAN2 [projector](https://github.com/NVlabs/stylegan2/blob/master/run_projector.py), please follow the **Requirements** in [ stylegan2 ](https://github.com/NVlabs/stylegan2).  

The corresponding **latent code** (in WP(W+) latent space) `{name}_wp.npy` should be placed in `DATA_PATH/code`. 

**Please find the examplar data in `./test`**

#### Run

For diffuse method:

```python
python main_diffuse.py --data_dir DATA_PATH  --boundary_path ./interface/boundaries/fine/all --alpha -5.0 --latent_space_type WP
```

The resulting images will be saved in DATA_PATH/diffuse_res, the resulting latent codes will be saved in DATA_PATH/diffuse_code

For warp method:

```python
python main_warp.py --data_dir DATA_PATH --boundary_path ./interface/boundaries/fine/all --alpha -5.0 --latent_space_type WP
```

The resulting images will be saved in DATA_PATH/warp_res.

### Training

#### coarse separation boundary training

1. Data generation:

   ```python
   python generate_data_and_score.py --output_dir DATASET_PATH --num DATASET_SIZE --truncation_psi 0.8
   ```
   

​	2.Coarse separation boundary training:

```python
python train_coarse_boundary.py --output_dir COARSE_BOUNDARY_DIR --latent_codes_path DATASET_PATH/w.npy  --scores_path DATASET_PATH/double_chin_scores.npy --chosen_num_or_ratio 0.1 --split_ratio 0.9 
```

The coarse separation boundary will be saved in `COARSE_BOUNDARY_DIR`.

You can also use the pretrained  coarse separation boundary in `./interface/boundaries/coarse/psi_0.8/stylegan2_ffhq_double_chin_w`

#### fine separation boundary training

1. First, **prepare data for diffusion**:

```python
python remove_double_chin_step1.py  --output_dir TRAINING_DIR --boundary_path COARSE_BOUNDARY_DIR  --input_data_dir DATASET_PATH
```
2. Then **diffuse the prepared data**:

```python
python remove_double_chin_step2.py --data_dir TRAINING_DIR
```

Resulting images of diffusion will be saved in  `TRAINING_DIR/diffused`, resulting latent codes will be saved in  `TRAINING_DIR/codes`.

3. After diffuse, you can use the results of diffuse to **train the fine separation boundary**:

```python
python train_fine_boundary.py --output_dir FINE_BOUNDARY_DIR --latent_codes_path TRAINING_DIR/codes --split_ratio 0.9
```

The coarse separation boundary will be saved in `FINE_BOUNDARY_DIR`

## Contact

onethousand@zju.edu.cn / [onethousand1250@gmail.com](mailto:onethousand1250@gmail.com)

## Citation

**If you use this code for your research, please cite our paper:**

```
@article{DBLP:journals/tog/WuYX021,
  author    = {Yiqian Wu and
               Yong{-}Liang Yang and
               Qinjie Xiao and
               Xiaogang Jin},
  title     = {Coarse-to-fine: facial structure editing of portrait images via latent
               space classifications},
  journal   = {{ACM} Trans. Graph.},
  volume    = {40},
  number    = {4},
  pages     = {46:1--46:13},
  year      = {2021}
}
```



## Reference and Acknowledgements

We thanks the following works:

[StyleGAN2](https://github.com/NVlabs/stylegan2)

[InterFaceGAN](https://github.com/genforce/interfacegan)

[StyleGAN2(pytorch-version)](https://github.com/Tetratrio/stylegan2_pytorch)

[face-alignment](https://github.com/1adrianb/face-alignment)

[idinvert](https://github.com/genforce/idinvert)

[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

```
@inproceedings{zhu2020indomain,
  title     = {In-domain GAN Inversion for Real Image Editing},
  author    = {Zhu, Jiapeng and Shen, Yujun and Zhao, Deli and Zhou, Bolei},
  booktitle = {Proceedings of European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
@inproceedings{shen2020interpreting,
  title     = {Interpreting the Latent Space of GANs for Semantic Face Editing},
  author    = {Shen, Yujun and Gu, Jinjin and Tang, Xiaoou and Zhou, Bolei},
  booktitle = {CVPR},
  year      = {2020}
}
@inproceedings{Karras2019stylegan2,
  title     = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author    = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. CVPR},
  year      = {2020}
}
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

