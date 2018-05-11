import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.layers import RepeatVector, Reshape, Concatenate, Conv2D, UpSampling2D, Conv2DTranspose, Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab

input_tensor = Input(shape=(224, 224, 1))
input_concat = Concatenate(axis=3)([input_tensor, input_tensor, input_tensor])
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), input_tensor=input_concat)
for layer in vgg16.layers:
    layer.trainable = False

upsample1 = UpSampling2D()(vgg16.get_layer('block5_pool').output)
deconv1 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(upsample1)
d1 = Dropout(0.2)(deconv1)
deconv2 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(d1)
bn1 = BatchNormalization()(deconv2)

upsample2 = UpSampling2D()(bn1)
deconv3 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(upsample2)
d2 = Dropout(0.2)(deconv3)
deconv4 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(d2)
bn2 = BatchNormalization()(deconv4)

upsample3 = UpSampling2D()(bn2)
deconv5 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(upsample3)
d3 = Dropout(0.2)(deconv5)
deconv6 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(d3)
bn3 = BatchNormalization()(deconv6)

upsample4 = UpSampling2D()(bn3)
deconv7 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(upsample4)
d4 = Dropout(0.2)(deconv7)
deconv8 = Conv2D(256, 3, padding='same', activation='relu')(deconv7)

upsample5 = UpSampling2D()(deconv8)
deconv9 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(upsample5)
deconv10 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(deconv9)
output = Conv2D(2, kernel_size=3, padding='same', activation='relu')(deconv10)
model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

def train_gen(batch_size):
    paths = os.listdir('./data/train/')
    while True:
        np.random.shuffle(paths)
        imgs = []
        for p in paths[:batch_size]:
            imgs.append(
                img_to_array(load_img('./data/train/' + p, target_size=(224, 224))))
        imgs = np.array(imgs, dtype=float)
        imgs = np.divide(imgs, 255)
        X_lab = rgb2lab(imgs)
        X_batch = X_lab[:, :, :, 0] / 100
        X_input = X_batch.reshape(X_batch.shape+(1,))
        X_output = X_lab[:, :, :, 1:] / 256
        X_output += 0.5
        yield X_input, X_output

def validation_gen(batch_size):
    paths = os.listdir('./data/validation/')
    while True:
        np.random.shuffle(paths)
        imgs = []
        for p in paths[:batch_size]:
            imgs.append(img_to_array(
                load_img('./data/validation/' + p, target_size=(224, 224))))
        imgs = np.array(imgs, dtype=float)
        imgs = np.divide(imgs, 255)
        X_lab = rgb2lab(imgs)
        X_batch = X_lab[:, :, :, 0] / 100
        X_input = X_batch.reshape(X_batch.shape+(1,))
        X_output = X_lab[:, :, :, 1:] / 256
        X_output += 0.5
        yield X_input, X_output

csv_logger = CSVLogger(filename='./VGG-AutoEncoder-logs.csv')
reduce_lr_on_plateau = ReduceLROnPlateau(patience=5, min_lr=1e-8, factor=0.5)

epochs = 50
batch_size = 32
model.fit_generator(train_gen(batch_size),
                    steps_per_epoch=1900 // batch_size,
                    callbacks=[csv_logger, reduce_lr_on_plateau],
                    shuffle=True,
                    validation_data=validation_gen(batch_size),
                    validation_steps=500 // batch_size,
                    epochs=epochs,
                    verbose=2)
model.save(filepath='./VGG-AutoEncoder-model.h5')
