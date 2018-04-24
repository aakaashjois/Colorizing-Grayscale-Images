import sys
import os
import shutil
import argparse
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
        save_directory = os.path.join('data', directory)
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory)
        os.mkdir(save_directory)
        path = os.path.join(raw_images_dir, directory)
        files_list = os.listdir(path)
        for file_name in tqdm(files_list):
            img = crop_and_resize_image(Image.open(os.path.join(path, file_name)), img_size)
            img.save(os.path.join(save_directory, file_name), 'jpeg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='Absolute path to directory of downloaded images. Should contain folders train, validation and test in it', type=str)
    parser.add_argument('-s', '--size', help='Size of image. The images will be resized to a square so just one width should be entered. Default - 244', type=int)
    args = parser.parse_args()
    if not os.path.exists(args.directory):
        raise ValueError('Invalid path to directory')
    raw_images_dir = args.directory
    img_size = args.size if args.size is not None else 244
    load_data(raw_images_dir, img_size)
