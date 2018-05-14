import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, RepeatVector, Reshape, Concatenate, UpSampling2D, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab


class ModelUtils:
    def __get_pretrained_model(self, model_type, include_top, freeze, input_tensor):
        if model_type == 'inception':
            model = InceptionV3(weights='imagenet',
                                include_top=include_top,
                                input_shape=(224, 224),
                                input_tensor=input_tensor)
        elif model_type == 'vgg':
            model = VGG16(weights='imagenet',
                          include_top=include_top,
                          input_shape=(224, 224),
                          input_tensor=input_tensor)
        else:
            raise ValueError('Unknown model type. Expected "inception" or "vgg".')
        if freeze:
            for layer in model.layers:
                layer.trainable = False
        return model

    def get_deep_koalarization_model(self):
        model_input = Input(shape=(224, 224, 1))
        encoder_output = Conv2D(64, 3, activation='relu', padding='same', strides=2)(model_input)
        encoder_output = Conv2D(128, 3, activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, 3, activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, 3, activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, 3, activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, 3, activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, 3, activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, 3, activation='relu', padding='same')(encoder_output)

        embedding_input = Concatenate(axis=3)([model_input, model_input, model_input])
        inception = self.__get_pretrained_model(model_type='inception',
                                                include_top=True,
                                                freeze=True,
                                                input_tensor=embedding_input)

        fusion_output = RepeatVector(28 * 28)(inception.output)
        fusion_output = Reshape(([28, 28, 1000]))(fusion_output)
        fusion_output = Concatenate(axis=3)([encoder_output, fusion_output])
        fusion_output = Conv2D(256, 1, activation='relu', padding='same')(fusion_output)

        decoder_output = Conv2D(128, 3, activation='relu', padding='same')(fusion_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, 3, activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D(3)(decoder_output)
        decoder_output = Conv2D(32, 3, activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, 3, activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, 3, activation='sigmoid', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)

        return Model(inputs=model_input, outputs=decoder_output)

    def get_inception_vgg_autoencoder(self):
        input_tensor = Input(shape=(224, 224, 1))
        model_input = Concatenate(axis=3)([input_tensor, input_tensor, input_tensor])
        inception = self.__get_pretrained_model(model_type='inception',
                                                include_top=True,
                                                freeze=True,
                                                input_tensor=model_input)
        vgg16 = self.__get_pretrained_model(model_type='vgg',
                                            include_top=False,
                                            freeze=True,
                                            input_tensor=model_input)

        repeat = RepeatVector(7 * 7)(inception.output)
        reshape = Reshape([7, 7, 1000])(repeat)
        fusion = Concatenate(axis=3)([vgg16.output, reshape])
        conv2d_f = Conv2D(512, kernel_size=1, activation='relu', padding='same')(fusion)

        upsample1 = UpSampling2D()(conv2d_f)
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
        deconv8 = Conv2D(256, 3, padding='same', activation='relu')(d4)

        upsample5 = UpSampling2D()(deconv8)
        deconv9 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(upsample5)
        deconv10 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(deconv9)
        output = Conv2D(2, kernel_size=3, padding='same', activation='relu')(deconv10)
        return Model(inputs=input_tensor, outputs=output)

    def get_vgg_autoencoder_model(self):
        input_tensor = Input(shape=(224, 224, 1))
        model_input = Concatenate(axis=3)([input_tensor, input_tensor, input_tensor])
        vgg16 = self.__get_pretrained_model(model_type='vgg',
                                            include_top=False,
                                            freeze=True,
                                            input_tensor=model_input)

        upsample1 = UpSampling2D()(vgg16.output)
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
        deconv8 = Conv2D(256, 3, padding='same', activation='relu')(d4)

        upsample5 = UpSampling2D()(deconv8)
        deconv9 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(upsample5)
        deconv10 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(deconv9)
        output = Conv2D(2, kernel_size=3, padding='same', activation='relu')(deconv10)
        return Model(inputs=input_tensor, outputs=output)

    def get_gan_model(self):

        def create_generator_model():
            input_tensor = Input(shape=(224, 224, 1))
            model_input = Concatenate(axis=3)([input_tensor, input_tensor, input_tensor])
            vgg16 = self.__get_pretrained_model(model_type='vgg',
                                                include_top=False,
                                                freeze=True,
                                                input_tensor=model_input)
            upsample1 = UpSampling2D()(vgg16.output)
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

        def create_discriminator_model():
            input_l = Input(shape=(224, 224, 1))
            input_ab = Input(shape=(224, 224, 2))
            model_input = Concatenate(axis=3)([input_l, input_ab])
            inception = self.__get_pretrained_model(model_type='inception',
                                                    include_top=False,
                                                    freeze=True,
                                                    input_tensor=model_input)
            global_average_pool = GlobalAveragePooling2D()(inception.get_layer(name='mixed10').output)
            dense1 = Dense(units=512, activation='relu')(global_average_pool)
            dense2 = Dense(units=128, activation='relu')(dense1)
            output = Dense(units=1, activation='sigmoid')(dense2)
            return Model(inputs=[input_l, input_ab], outputs=output)

        def create_gan_model(input_tensor, generator, discriminator):
            generated_images = generator(inputs=input_tensor)
            predicted_labels = discriminator(inputs=[input_tensor, generated_images])
            return Model(inputs=input_tensor, outputs=[generated_images, predicted_labels])

        input_tensor = Input(shape=(224, 224, 1))
        generator_model = create_generator_model()
        discriminator_model = create_discriminator_model()
        gan_model = create_gan_model(input_tensor, generator_model, discriminator_model)
        return generator_model, discriminator_model, gan_model


class DatasetUtils:
    def __init__(self):
        self.image_generator = ImageDataGenerator()

    def get_image_generator(self, generator_type, batch_size):
        if generator_type == 'train':
            path = '../data/train'
        elif generator_type == 'validation':
            path = '../data/validation'
        elif generator_type == 'test':
            path = '../data/test'
        else:
            raise ValueError('Unknown type. Expected "train", "validation", or "test".')
        for batch in self.image_generator.flow_from_directory(directory=path,
                                                              target_size=(224, 224),
                                                              batch_size=batch_size,
                                                              class_mode=None):
            batch = np.divide(batch, 255)
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:, :, :, 0] / 100
            X_batch = X_batch.reshape(X_batch.shape + (1,))
            y_batch = (lab_batch[:, :, :, 1:] / 256) + 0.5
            yield X_batch, y_batch
