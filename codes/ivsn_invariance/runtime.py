"""Runtime for IVSN invariance experiments."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import math
import numpy as np
import random


ALT_CATEGORY_NAMES = {'teddybears': ['teddybears', 'teddy_bears']}


CATEGORIES_6 = ['sheep', 'cattle', 'cats', 'horses', 'teddybears', 'kites']


CATEGORIES_8 = ['sheep', 'cattle', 'cats', 'horses', 'teddybears', 'kites', 'dogs', 'elephants']


IMAGE_SIZE = 720


OBJ_SIZE = 156


RADIUS_6 = 220


RADIUS_8 = 240


CATEGORIES = None


N_POSITIONS = None


RADIUS = None


POSITIONS = None


MAX_FIXATIONS = None


DEFAULT_N_IDENTICAL = 150


DEFAULT_N_DIFFERENT = 100


EARLY_SUCCESS_FIXATIONS = 3


ORACLE_WINDOW = 45


SEED = 0


DEFAULT_GIST_CONFIG = dict(in_channels=1, mode='dynamic', fmax=0.35, fratio=1.7, k=0.52, n_scales=4, n_orientations=6, n_phases=2, scale=25, gaussian=True, gaussian_inverse=False, n_stds=3, dc_compensate=True, stride=4, energy=True, energy_mode='substitute', divisive_norm=False, pooling=None, pool_size=16, pool_stride=16, flatten=False)


CODES_DIR = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS_DIR = CODES_DIR / 'model_weights'

DEFAULT_GIST_CHECKPOINTS = {
    'vgg_gist_pretrained': MODEL_WEIGHTS_DIR / 'vgg_gist_model_epoch_25.pth',
    'conv_gist': MODEL_WEIGHTS_DIR / 'conv_gist_model_epoch_15.pth',
    'conv_gist_mlp': MODEL_WEIGHTS_DIR / 'conv_gist_mlp_model_epoch_10.pth',
    'vgg_gist_imagenet64': MODEL_WEIGHTS_DIR / 'vgg_gist_imagenet64_epoch25.pth',
}


def get_categories(n_objects: int):
    if n_objects == 6:
        return CATEGORIES_6
    if n_objects == 8:
        return CATEGORIES_8
    raise ValueError(f'Unsupported n_objects: {n_objects}')


def circle_positions(image_size: int, radius: int, n: int):
    center = image_size // 2
    pts = []
    start_angle = -math.pi / 2.0
    for i in range(n):
        angle = start_angle + 2 * math.pi * i / n
        x = center + int(round(math.cos(angle) * radius))
        y = center + int(round(math.sin(angle) * radius))
        pts.append((x, y))
    return pts


def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_font(size=18):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except Exception:
        return ImageFont.load_default()


def set_runtime_geometry(n_objects: int):
    global CATEGORIES, N_POSITIONS, RADIUS, POSITIONS, MAX_FIXATIONS
    CATEGORIES = get_categories(n_objects)
    N_POSITIONS = n_objects
    RADIUS = RADIUS_6 if n_objects == 6 else RADIUS_8
    POSITIONS = circle_positions(IMAGE_SIZE, RADIUS, N_POSITIONS)
    MAX_FIXATIONS = n_objects
