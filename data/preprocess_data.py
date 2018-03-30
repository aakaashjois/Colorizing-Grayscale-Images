import sys
import os
import shutil
import numpy as np

from tqdm import tqdm
from PIL import Image


def crop_and_resize_image(img, size):
    width, height = img.size
    smaller_dim = width if width <= height else height
    w_diff = np.floor((width - smaller_dim) / 2)
    h_diff = np.floor((height - smaller_dim) / 2)
    left = w_diff
    right = w_diff + smaller_dim
    top = h_diff
    bottom = h_diff + smaller_dim
    cropped = img.crop((left, top, right, bottom))
    resized = img.resize((size, size), Image.LANCZOS)
    return resized


def load_data(raw_images_dir, img_size):
    dirs = os.listdir(raw_images_dir)
    for directory in dirs:
        print('Preprocessing {}:'.format(directory))
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        path = os.path.join(raw_images_dir, directory)
        files_list = os.listdir(path)
        for file_name in tqdm(files_list):
            img = crop_and_resize_image(Image.open(os.path.join(path, file_name)), img_size)
            img.save(os.path.join(directory, file_name), 'jpeg')


if __name__ == "__main__":
    files = os.listdir(os.getcwd())
    if 'preprocess_data.py' not in files:
        sys.exit('Please run the script with data directory as the root')
    raw_images_dir = input('Enter path for raw images directory: ')
    if raw_images_dir is None:
        sys.exit('Invalid data entered')
    if not os.path.exists(raw_images_dir):
        sys.exit('Invalid path')
    img_size = int(input('Enter image size to resize: '))
    if img_size is None:
        sys.exit('Invalid data entered')
    load_data(raw_images_dir, img_size)
