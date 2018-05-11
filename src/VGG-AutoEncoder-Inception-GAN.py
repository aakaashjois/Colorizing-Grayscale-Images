import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import RepeatVector, Reshape, Concatenate, Conv2D, UpSampling2D, Conv2DTranspose, Input, \
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.losses import mean_squared_error, binary_crossentropy
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab


def get_generator():
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
    return Model(inputs=input_tensor, outputs=output)


def get_discriminator():
    input_l = Input(shape=(224, 224, 1))
    input_ab = Input(shape=(224, 224, 2))
    input_lab = Concatenate(axis=3)([input_l, input_ab])
    inception = InceptionV3(include_top=False, input_shape=(224, 224, 3), weights='imagenet', input_tensor=input_lab)
    for layer in inception.layers:
        layer.trainable = False
    global_average_pool = GlobalAveragePooling2D()(inception.get_layer(name='mixed10').output)
    dense1 = Dense(units=512, activation='relu')(global_average_pool)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=1, activation='sigmoid')(dense2)
    return Model(inputs=[input_l, input_ab], outputs=output)


def get_gan(input_tensor, g, d):
    generated_images = g(inputs=input_tensor)
    predicted_labels = d(inputs=[input_tensor, generated_images])
    return Model(inputs=gan_input, outputs=[generated_images, predicted_labels])


def image_gen(list_paths, dir_path, batch_size):
    while True:
        img_batch = []
        np.random.shuffle(list_paths)
        for path in list_paths[:batch_size]:
            img_rgb = img_to_array(load_img(dir_path + path, target_size=(224, 224)))
            img_batch.append(img_rgb)

        img_rgb_batch = np.array(img_batch, dtype=float)
        img_rgb_batch = np.divide(img_rgb_batch, 255)
        img_lab_batch = rgb2lab(img_rgb_batch)
        img_l_batch = img_lab_batch[:, :, :, 0] / 100
        img_l_batch = img_l_batch.reshape(img_l_batch.shape + (1,))
        img_ab_batch = (img_lab_batch[:, :, :, 1:] / 256) + 0.5
        yield img_l_batch, img_ab_batch


gan_input = Input(shape=(224, 224, 1))
generator = get_generator()
discriminator = get_discriminator()
gan = get_gan(gan_input, generator, discriminator)

discriminator.trainable = True
discriminator.compile(optimizer='adam', loss=binary_crossentropy)
discriminator.trainable = False
gan.compile(optimizer='adam', loss=[mean_squared_error, binary_crossentropy])
discriminator.trainable = True
generator.compile(optimizer='adam', loss=[mean_squared_error])

batch_size = 32
train_path = './data/train/'
validation_path ='./data/validation/'
train_gen = image_gen(os.listdir(train_path), train_path, batch_size)
validation_gen = image_gen(os.listdir(validation_path), validation_path, len(os.listdir(validation_path)))
y_true_batch = np.ones(batch_size)
y_pred_batch = np.zeros(batch_size)

epochs = 50
steps = 1900
d_loss_per_epoch = []
g_loss_per_epoch = []
val_loss_per_epoch = []
for epoch in range(epochs):
    d_losses = []
    g_losses = []
    for step in range(steps):
        train_l, train_ab = next(train_gen)
        gen_pred = generator.predict_on_batch(train_l)
        d_loss_true = discriminator.train_on_batch([train_l, train_ab], y_true_batch)
        d_loss_pred = discriminator.train_on_batch([train_l, gen_pred], y_pred_batch)
        d_loss = (d_loss_true + d_loss_pred) / 2
        d_losses.append(d_loss)
        discriminator.trainable = False
        gan_loss = gan.train_on_batch(train_l, [train_ab, y_true_batch])
        g_losses.append(gan_loss)
        discriminator.trainable = True

    val_l, val_ab = next(validation_gen)
    val_loss = generator.test_on_batch(x=val_l, y=val_ab)
    val_loss_per_epoch.append(val_loss)
    print('Epoch: {} - d_loss: {} - g_loss: {} - val_loss: {}'.format(epoch, np.mean(d_losses), np.mean(g_losses), np.mean(val_losses)))
    d_loss_per_epoch.append(np.mean(d_losses))
    g_loss_per_epoch.append(np.mean(g_losses))

gan.save('./GAN-Models/gan-model.h5')
generator.save('./GAN-Models/generator-model.h5')
discriminator.save('./GAN-Models/discriminator-model.h5')
