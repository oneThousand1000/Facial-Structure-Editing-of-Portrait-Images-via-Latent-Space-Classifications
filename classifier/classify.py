import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from .src.config import Config
from .src.classifier import Classifier

def get_model(mode=None):
    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = Classifier(config)
    model.load()
    return  model



def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    # # parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
    #
    # # test mode
    # if mode == 2:
    #     parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    #     parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    #     parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
    #     parser.add_argument('--output', type=str, help='path to the output directory')

    #args = parser.parse_args()



    config_path = os.path.join(os.path.dirname(__file__),'./config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    # if mode == 1:
    #     config.MODE = 1
    #     if args.model:
    #         config.MODEL = args.model

    # test mode
    # elif mode == 2:
    #     config.MODE = 2
    #     config.MODEL = args.model if args.model is not None else 3
    #     config.INPUT_SIZE = 0
    #
    #     if args.input is not None:
    #         config.TEST_FLIST = args.input
    #
    #     if args.mask is not None:
    #         config.TEST_MASK_FLIST = args.mask
    #
    #     if args.edge is not None:
    #         config.TEST_EDGE_FLIST = args.edge
    #
    #     if args.output is not None:
    #         config.RESULTS = args.output

    # eval mode
    # elif mode == 3:
    #     config.MODE = 3
    #     config.MODEL = args.model if args.model is not None else 3

    return config

def check_double_chin(img,model):
    output= model.process(img)
    return output[0][1]>0.95


