import os
from torch import device, cuda

MODEL_DIR = os.path.abspath('model-registry')
SUPPORT_PATH = os.path.abspath('batik')

IMAGE_SIZE = 84
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TEST_CLASS = ["Batik Tambal", "Batik Megamendung", "Batik Insang", "Batik Bali", "Batik Ikat Celup", "Batik Sekar Jagad"]
TRAIN_CLASS = ["Batik Pala", "Batik Kawung", "Batik Parang", "Batik Geblek Renteng", "Batik Poleng", "Batik Dayak", "Batik Betawi", "Batik Lasem", "Batik Cendrawasih"]

BACKEND = ['eager', 'inductor']

DEVICE = device('cuda' if cuda.is_available() else 'cpu')