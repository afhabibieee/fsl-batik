import os
from torch import device, cuda

BATIK_SPECS_DIR = os.path.abspath('../../data/batik')

IMAGE_SIZE = 84
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_WAY = 3
N_SHOT = 5
N_QUERY = 5
N_TASK = 200
N_TRAINING_EPISODES = 500
N_VALIDATION_TASK = 100
N_WORKERS = 2

DEVICE = device('cuda' if cuda.is_available() else 'cpu')