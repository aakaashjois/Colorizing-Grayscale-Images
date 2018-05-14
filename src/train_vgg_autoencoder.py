from keras.callbacks import CSVLogger

from src import utils

# Obtain the model from utilities
model = utils.ModelUtils().get_vgg_autoencoder_model()
model.compile(optimizer='adam', loss='mse')

# Create callback to log the training history
csv_logger = CSVLogger(filename='./VGG-AutoEncoder-logs.csv')

epochs = 50
batch_size = 32

# Obtain the image generators from utilities
dataset_utils = utils.DatasetUtils()
train_gen = dataset_utils.get_image_generator(generator_type='train', batch_size=batch_size)
validation_gen = dataset_utils.get_image_generator(generator_type='validation', batch_size=batch_size)

# Train model using generator functions
model.fit_generator(train_gen,
                    callbacks=[csv_logger],
                    shuffle=True,
                    validation_data=validation_gen,
                    epochs=epochs,
                    verbose=2)

# Save the model
model.save(filepath='./VGG-AutoEncoder-model.h5')
