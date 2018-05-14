import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave

path_to_model = '' # Update path before running
path_to_results = '' # Update path before running

test_images = []
for test_path in os.listdir(path='./data/test/'):
    test_images.append(img_to_array(load_img('./data/test/' + test_path, target_size=(224, 224))))
test_images = np.array(test_images, dtype=float)
test_images = np.divide(test_images, 255)
test_images = rgb2lab(test_images)
test_l = test_images[:, :, :, 0] / 100
test_l = test_l.reshape(test_l.shape + (1,))

model = load_model(path_to_model)

predictions = model.predict(x=test_l, batch_size=1)

lab_predictions = (predictions - 0.5) * 256
lab_results = np.concatenate(((test_l * 100), lab_predictions), axis=3)

rgb_results = []
for i, result in enumerate(lab_results):
    rgb_results.append(lab2rgb(result))
    imsave(path_to_results + 'img_{}.png'.format(i), lab2rgb(result))