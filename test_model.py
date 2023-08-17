import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append('.')
from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args

class DotDict:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        if key in self.data:
            value = self.data[key]
            if isinstance(value, dict):
                return DotDict(value)
            else:
                return value
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

imgA = Image.open("data/MT-Dataset/images/non-makeup/vSYYZ113.png").convert('RGB') # change with the non-makeup file
imgB = Image.open("data/MT-Dataset/images/makeup/vFG474.png").convert('RGB') # change with the makeup file
save_path = "result1.png"

data ={"name":"demo","save_path":"result","load_path":"ckpts/sow_pyramid_a5_e3d2_remapped.pth","source_dir":"data/MT-Dataset/images/non-makeup","reference_dir":"data/MT-Dataset/images/makeup","gpu":"cuda:0","device":torch.device("cuda:0"), "save_folder":"result/demo"}
args = DotDict(data)
config = get_config()
inference = Inference(config, args, args.load_path)
result = inference.transfer(imgA, imgB, postprocess=True)
imgA = np.array(imgA)
h, w, _ = imgA.shape
result2 = result.resize((h, w)); result2 = np.array(result)
# vis_image = np.hstack((result))
Image.fromarray(result2.astype(np.uint8)).save(save_path)