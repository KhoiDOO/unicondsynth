from diffusers import StableDiffusionInstructPix2PixPipeline

from PIL import Image
from utils import *

import numpy as np
import torch
import os
import random

if __name__ == '__main__':

    get_args()

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16)