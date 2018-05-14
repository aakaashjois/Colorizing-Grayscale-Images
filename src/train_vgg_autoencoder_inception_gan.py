import numpy as np

from src import utils

# Obtain the models from utilities function
generator, discriminator, gan = utils.ModelUtils().get_gan_model()
# Make the discriminator trainable and compile it
discriminator.trainable = True
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
# Make the discriminator not trainable and compile the complete GAN model
discriminator.trainable = False
gan.compile(optimizer='adam', loss=['mse', 'binary_crossentropy'])
# Make the discriminator trainable again and compile only the generator model
discriminator.trainable = True
generator.compile(optimizer='adam', loss='mse')

batch_size = 32

# Obtain the generator functions from utilities
dataset_utils = utils.DatasetUtils()
train_gen = dataset_utils.get_image_generator(generator_type='train', batch_size=batch_size)
validation_gen = dataset_utils.get_image_generator(generator_type='validation', batch_size=batch_size)

# Create an array of labels for ground truth images and generated images
y_true_batch = np.ones(batch_size)
y_pred_batch = np.zeros(batch_size)

epochs = 50
train_steps = len(train_gen) // batch_size
validation_steps = len(validation_gen) // batch_size

d_loss_per_epoch = []
g_loss_per_epoch = []
val_loss_per_epoch = []

for epoch in range(epochs):
    d_losses = []
    g_losses = []
    val_losses = []

    train_gen = dataset_utils.get_image_generator(generator_type='train', batch_size=batch_size)
    validation_gen = dataset_utils.get_image_generator(generator_type='validation', batch_size=batch_size)

    for step in range(train_steps):
        train_l, train_ab = next(train_gen)
        # Get initial predictions from generator
        gen_pred = generator.predict_on_batch(train_l)
        # Train discriminator on ground truth and generator outputs
        d_loss_true = discriminator.train_on_batch([train_l, train_ab], y_true_batch)
        d_loss_pred = discriminator.train_on_batch([train_l, gen_pred], y_pred_batch)
        d_loss = (d_loss_true + d_loss_pred) / 2
        d_losses.append(d_loss)
        # Freeze discriminator and train the GAN on ground truth
        discriminator.trainable = False
        gan_loss = gan.train_on_batch(train_l, [train_ab, y_true_batch])
        g_losses.append(gan_loss)
        discriminator.trainable = True

    for step in range(validation_steps):
        val_l, val_ab = next(validation_gen)
        # Test generator on validation images
        val_loss = generator.test_on_batch(x=val_l, y=val_ab)
        val_losses.append(val_loss)

    print('Epoch: {} - d_loss: {} - g_loss: {} - val_loss: {}'.format(epoch, np.mean(d_losses), np.mean(g_losses),
                                                                      np.mean(val_losses)))
    d_loss_per_epoch.append(np.mean(d_losses))
    g_loss_per_epoch.append(np.mean(g_losses))
    val_loss_per_epoch.append(np.mean(val_losses))

# Save the models
gan.save('./gan-model.h5')
generator.save('./generator-model.h5')
discriminator.save('./discriminator-model.h5')
