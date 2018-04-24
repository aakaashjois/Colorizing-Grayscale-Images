import create_model
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from os import listdir
import time

keras = tf.keras
Model = keras.models.Model

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

d.trainable = True
d.compile(optimizer='adam', loss=wasserstein_loss)
d.trainable = False
gan.compile(optimizer='adam', loss=[perceptual_loss, wasserstein_loss], loss_weights=[100, 1])
d.trainable = True

train_paths = np.array(listdir('./data/train/'))
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
        img_np_noise = np.copy(img_np)
        if len(img_np.shape) == 3:
            img_np_noise[:,:,1] = np.random.random(img_np.shape[:2])
            img_np_noise[:,:,2] = np.random.random(img_np.shape[:2])

            x_train.append(img_np)
            x_train_noise.append(img_np_noise)

stop = time.time()
print(stop - start, flush=True)

x_train = np.array(x_train)
x_train_noise = np.array(x_train_noise)

epochs = 100
steps = 1850
batch_size = 64
batch_start_index = 0

for e in tqdm(range(epochs)):
    # Seed the generator to shuffle ground truth and noise in the same manner
    # Shuffle once per epoch
    seed = np.random.randint(10e5)
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(x_train_noise)
    
    d_losses = []
    gan_losses = []
    
    for s in range(steps):
        # Seed the generator to randomly pick ground truth and noise in the same manner
        seed = np.random.randint(10e5)
        np.random.seed(seed)
        x_train_batch = x_train[np.random.choice(len(x_train), size=batch_size)]
        np.random.seed(seed)
        x_train_noise_batch = x_train_noise[np.random.choice(len(x_train_noise), batch_size)]
        np.random.seed()
        
        # Get fake color predictions from generator
        g_pred = g.predict(x_train_batch, batch_size=batch_size)
        
        for _ in range(5):
            # Real and fake labels
            y_train_true = np.ones(batch_size)
            y_train_fake = np.zeros(batch_size)
            
            # Real and fake losses; train discriminator on ground truth-true labels and noise-fake labels
            d_loss_true = d.train_on_batch(x_train_batch, y_train_true)
            d_loss_fake = d.train_on_batch(g_pred, y_train_fake)
            d_losses.append(0.5 * (d_loss_true + d_loss_fake))
        
        print('Epoch {} batch {} d_loss {}'.format(e, s, np.mean(d_losses)), flush=True)
        
        # Freeze discriminator because we want to train the GAN as a whole
        d.trainable = False
        
        # Train GAN so that the generator learns to generate better colors
        gan_loss = gan.train_on_batch(x_train_noise_batch, [x_train_batch, y_true_train])
        gan_losses.append(gan_loss)
        
        print('Epoch {} batch {} gan_loss {}'.format(e, s, np.mean(gan_losses)), flush=True)
        
        # Un-freeze discriminator
        d.trainable = True
    
g.save('./models/g.h5')
d.save('./models/d.h5')
gan.save('./models/gan.h5')
