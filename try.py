import numpy as np
from utils import *
import os

file_names = os.listdir(args.gaze_path)

mat = input_generator(file_names, 500, 25, 'HR', 50)

print mat
