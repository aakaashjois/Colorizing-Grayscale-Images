from keras.callbacks import CSVLogger, ReduceLROnPlateau

from src import utils

model = utils.ModelUtils().get_inception_vgg_autoencoder()
model.compile(optimizer='adam', loss='mse')

csv_logger = CSVLogger(filename='./Inception-VGG-AutoEncoder-logs.csv')
reduce_lr_on_plateau = ReduceLROnPlateau(patience=5, min_lr=1e-8, factor=0.5)

epochs = 50
batch_size = 32

dataset_utils = utils.DatasetUtils()
train_gen = dataset_utils.get_image_generator(generator_type='train', batch_size=batch_size)
validation_gen = dataset_utils.get_image_generator(generator_type='validation', batch_size=batch_size)

model.fit_generator(train_gen,
                    callbacks=[csv_logger, reduce_lr_on_plateau],
                    shuffle=True,
                    validation_data=validation_gen,
                    epochs=epochs,
                    verbose=2)

model.save(filepath='./Inception-VGG-AutoEncoder-model.h5')
