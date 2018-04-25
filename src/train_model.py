import create_model
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from os import listdir
import time

keras = tf.keras
Model = keras.models.Model
Adam = keras.optimizer.Adam

input_shape = (224, 224, 3)
input_layer = create_model.get_input_layer(input_shape)
g = create_model.get_generator_model(input_layer, input_shape)
d = create_model.get_discriminator_model(input_shape)
gan = create_model.get_gan_model(input_shape, g, d)

def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)

def perceptual_loss(y_true, y_pred):
    vgg16 = create_model.get_vgg16_model(input_shape)
    loss_model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return keras.backend.mean(keras.backend.square(loss_model(y_true) - loss_model(y_pred)))

def save_trail_images(e, truth, generated):
    for index, images in enumerate(zip(truth, generated)):
        truth_image, generated_image = images
        truth_image = Image.fromarray(np.uint8(truth_image * 255))
        truth_image = truth_image.convert('RGB')
        truth_image.save('./images/truth_' + str(index) + '_' + str(e) + 'e.jpg', 'JPEG')
        generated_image = Image.fromarray(np.uint8(generated_image * 255))
        generated_image = generated_image.convert('RGB')
        generated_image.save('./images/generated_' + str(index) + '_' + str(e) + 'e.jpg', 'JPEG')


d.trainable = True
d_opt = Adam(lr=1E-4, epsilon=1e-8)
gan_opt = Adam(lr=1E-4, epsilon=1e-8)
d.compile(optimizer=d_opt, loss=wasserstein_loss)
d.trainable = False
gan.compile(optimizer=gan_opt, loss=[perceptual_loss, wasserstein_loss], loss_weights=[10, 1])
d.trainable = True

train_paths = np.array(listdir('./data/train/'))

# Reducing number of images to train
train_paths = train_paths[0:12000]

_ = ['./data/train/'] * len(train_paths)
train_paths = np.core.defchararray.add(_,  train_paths)

# test_paths = listdir('./data/test/')
# _ = ['./data/test/'] * len(test_paths)
# test_paths = np.core.defchararray.add(_,  test_paths)

# validation_paths = listdir('./data/validation/')
# _ = ['./data/validation/'] * len(validation_paths)
# validation_paths = np.core.defchararray.add(_,  validation_paths)

start = time.time()
x_train = []
x_train_noise = []
for i in tqdm(train_paths):
    with Image.open(i) as img:
        img_np = np.array(img)
        # Normalize the image
        img_np = img_np - np.min(img_np) / (np.max(img_np) - np.min(img_np))
        img_np_noise = np.copy(img_np)
        # Use only the color images
        if len(img_np.shape) == 3:
            img_np_noise[:,:,1] = np.random.random(img_np.shape[:2])
            img_np_noise[:,:,2] = np.random.random(img_np.shape[:2])

            x_train.append(img_np)
            x_train_noise.append(img_np_noise)

stop = time.time()
print('Loading data took {} seconds'.format(stop - start), flush=True)

x_train = np.array(x_train)
x_train_noise = np.array(x_train_noise)

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

    trail_truth_images = x_train[0:10]
    trial_generated_images = g.predict(x_train_noise[0:10], batch_size=10)
    save_trail_images(e, trail_truth_images, trial_generated_images)

g.save('./models/g-{}e-{}.h5')
d.save('./models/d-{}e-{}.h5.h5')
gan.save('./models/gan-{}e-{}.h5.h5')
