import os

import cv2
from tqdm import tqdm
import numpy as np


def make_1mil_png(src, img_size):
    files_list = os.listdir(src)
    file_choices = np.random.randint(0, 1000000, 1000000)
    os.mkdir('train')
    os.mkdir('validate')
    os.mkdir('test')
    naming_counter = 0
    for i in tqdm(file_choices):
        if naming_counter < 700000:
            dst = 'train'
        elif naming_counter < 900000:
            dst = 'validate'
        else:
            dst = 'test'
        img = cv2.imread(os.path.join(src, files_list[i]))
        smaller_dim = img.shape[0] if img.shape[0] <= img.shape[1] else img.shape[1]
        img_crop = img[int(np.floor((img.shape[1] - smaller_dim)/2)): int(np.floor((img.shape[1] + smaller_dim)/2)), 
            int(np.floor((img.shape[0] - smaller_dim)/2)): int(np.floor((img.shape[0] + smaller_dim)/2))]
        cv2.resize(img_crop, [img_shape, img_shape])
        dst_img_path = os.path.join(dst, files_list[i])
        cv2.imwrite(dst_img_path, img)
        naming_counter += 1


if __name__ == "__main__":
    make_1mil_png('.\\bedroom_webp_flat')

