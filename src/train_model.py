import time
from os import listdir

import create_model
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

keras = tf.keras
Model = keras.models.Model
Adam = keras.optimizers.Adam
mean_squared_error = keras.losses.mean_squared_error
binary_crossentropy = keras.losses.binary_crossentropy

input_shape = (224, 224, 3)
input_layer = create_model.get_input_layer(input_shape)
g = create_model.get_generator_model(input_layer, input_shape)
d = create_model.get_discriminator_model(input_shape)
gan = create_model.get_gan_model(input_shape, g, d)


def save_images(e, truth, generated):
    for index, images in enumerate(zip(truth, generated)):
        truth_image, generated_image = images

        truth_image = cv2.cvtColor(np.uint8(truth_image * 255), cv2.COLOR_YUV2BGR)
        truth_image.save('./images/truth_' + str(index) + '_' + str(e) + 'e.jpg', 'JPEG')

        generated_image = cv2.cvtColor(np.uint8(generated_image * 255), cv2.COLOR_YUV2BGR)
        generated_image.save('./images/generated_' + str(index) + '_' + str(e) + 'e.jpg', 'JPEG')


def load_images(paths):
    start = time.time()
    images = []
    images_noise = []

    for path in tqdm(paths):
        img = cv2.imread(path)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_noise = np.copy(img)
        img_noise[:, :, 1] = np.random.random(img.shape[:2])
        img_noise[:, :, 2] = np.random.random(img.shape[:2])
        images.append(img)
        images_noise.append(img_noise)

    stop = time.time()
    print('Loading data took {} seconds'.format(stop - start), flush=True)

    return np.array(images), np.array(images_noise)


def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)


def perceptual_loss(y_true, y_pred):
    vgg16 = create_model.get_vgg16_model(input_shape)
    loss_model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return keras.backend.mean(keras.backend.square(loss_model(y_true) - loss_model(y_pred)))


def d_loss_fn(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def g_loss_fn(y_true, y_pred):
    p_loss = perceptual_loss(y_true, y_pred)
    mse_loss = mean_squared_error(y_true, y_pred)
    return 10 * p_loss + mse_loss


d.trainable = True
d_opt = Adam(lr=1E-4, epsilon=1e-8)
gan_opt = Adam(lr=1E-4, epsilon=1e-8)
d.compile(optimizer=d_opt, loss=d_loss_fn)
d.trainable = False
gan.compile(optimizer=gan_opt, loss=[g_loss_fn, d_loss_fn])
d.trainable = True

train_paths = np.array(listdir('./data/train/'))
# Reducing number of images to train
train_paths = train_paths[0:12000]
train_paths = np.core.defchararray.add(['./data/train/'] * len(train_paths), train_paths)

test_paths = listdir('./data/test/')
# Reducing number of images to test
train_paths = train_paths[0:10]
test_paths = np.core.defchararray.add(['./data/test/'] * len(test_paths),  test_paths)

# validation_paths = listdir('./data/validation/')
# validation_paths = np.core.defchararray.add(['./data/validation/'] * len(validation_paths),  validation_paths)

x_train, x_train_noise = load_images(train_paths)
x_test, x_test_noise = load_images(test_paths)

epochs = 10
steps = 5
batch_size = 256
batch_start_index = 0

# Real and fake labels
y_train_true = np.ones(batch_size)
y_train_fake = np.zeros(batch_size)

for e in tqdm(range(epochs)):
    # Seed the generator to shuffle ground truth and noise in the same manner
    # Shuffle once per epoch
    random_shuffle = np.random.permutation(len(x_train))
    x_train = x_train[random_shuffle]
    x_train_noise = x_train_noise[random_shuffle]
    d_losses = []
    gan_losses = []

    for s in range(steps):
        random_idx = np.random.choice(len(x_train), size=batch_size)
        x_train_batch = x_train[random_idx]
        x_train_noise_batch = x_train_noise[random_idx]

        # Get fake color predictions from generator
        g_pred = g.predict(x_train_batch, batch_size=batch_size)

        print('Train discriminator', flush=True)
        for _ in range(1):
            # Real and fake losses; train discriminator on ground truth-true labels and noise-fake labels
            d_loss_true = d.train_on_batch(x_train_batch, y_train_true)
            d_loss_fake = d.train_on_batch(g_pred, y_train_fake)
            d_losses.append(0.5 * (d_loss_true + d_loss_fake))

        print('Epoch {} - step {} - d_loss {}'.format(e, s, np.mean(d_losses)), flush=True)

        # Freeze discriminator because we want to train the GAN as a whole
        d.trainable = False

        # Train GAN so that the generator learns to generate better colors
        print('Train generator', flush=True)
        gan_loss = gan.train_on_batch(x_train_noise_batch, [x_train_batch, y_train_true])
        gan_losses.append(gan_loss)

        print('Epoch {} - step {} - gan_loss {}'.format(e, s, np.mean(gan_losses)), flush=True)

        # Un-freeze discriminator
        d.trainable = True

    test_generated_images = g.predict(x_test_noise, batch_size=10)
    save_images(e, x_test, test_generated_images)

save_time = time.time()
g.save('./models/g-{}e-{}.h5'.format(epochs, save_time))
d.save('./models/d-{}e-{}.h5'.format(epochs, save_time))
gan.save('./models/gan-{}e-{}.h5'.format(epochs, save_time))
