import tensorflow as tf


keras = tf.keras
Input = keras.layers.Input
BatchNormalization = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
UpSampling2D = keras.layers.UpSampling2D
Add = keras.layers.Add
Model = keras.models.Model
Dense = keras.layers.Dense
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
MaxPooling2D = keras.layers.MaxPool2D

def get_vgg16_model(input_shape, input_layer):
    """Returns a pretrained VGG16 model without the top layers
    
    Arguments:
        input_shape {tuple(int)} -- The shape of input layer
        input_layer {tensorflow.Tensor} -- The input layer for the VGG16 model
    
    Returns:
        tensorflow.keras.Model -- The VGG16 model pretrained on imagenet without top layers
    """

    return tf.keras.applications.vgg16.VGG16(include_top=False, 
                                             weights='imagenet',
                                             input_shape=input_shape, 
                                             input_tensor=input_layer)

def get_input_layer(input_shape):
    """Returns the input layer for the GAN model
    
    Arguments:
        input_shape {tuple(int)} -- The shape of input layer
    
    Returns:
        tensorflow.Tensor -- The input layer for the GAN model
    """

    return tf.keras.layers.Input(shape=input_shape)

def get_generator_model(input_layer, input_shape, train_vgg16_layers=False):
    """Returns the generator model based on the VGG16 architecture pretrained on imagenet
    
    Arguments:
        input_layer {tensorflow.Tensor} -- The input layer for the generator model
        input_shape {tuple(int)} -- The shape of input layer
        train_vgg16_layers {boolean} -- Whether to train VGG16 pretrained layers
    
    Returns:
        tensorflow.keras.Model -- The generator model
    """

    vgg16 = get_vgg16_model(input_shape, input_layer)
    vgg_layers = {layer.name: layer for layer in vgg16.layers}

    block5_bn = BatchNormalization()(vgg_layers['block5_pool'].output)
    block5_bn_conv = Conv2D(filters=512, kernel_size=1, activation='relu', padding='same')(block5_bn)
    block5_up = UpSampling2D(size=(2, 2))(block5_bn_conv)

    block4_bn = BatchNormalization()(vgg_layers['block4_pool'].output)
    block45_add = Add()([block4_bn, block5_up])
    block45_add_conv = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(block45_add)
    block45_up = UpSampling2D(size=(2, 2))(block45_add_conv)

    block3_bn = BatchNormalization()(vgg_layers['block3_pool'].output)
    block34_add = Add()([block3_bn, block45_up])
    block34_add_conv = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(block34_add)
    block34_up = UpSampling2D(size=(2, 2))(block34_add_conv)

    block2_bn = BatchNormalization()(vgg_layers['block2_pool'].output)
    block23_add = Add()([block2_bn, block34_up])
    block23_add_conv = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(block23_add)
    block23_up = UpSampling2D(size=(2, 2))(block23_add_conv)

    block1_bn = BatchNormalization()(vgg_layers['block1_pool'].output)
    block12_add = Add()([block1_bn, block23_up])
    block12_add_conv = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(block12_add)

    block12_up = UpSampling2D(size=(2, 2))(block12_add_conv)
    output = Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='same')(block12_up)

    model =  Model(inputs=input_layer, outputs=output, name='generator')

    if not train_vgg16_layers:
        all_layers = {layer.name: layer for layer in model.layers}
        for name, _ in all_layers.items():
            all_layers[name].trainable = False
        
    return model

def get_discriminator_model(input_shape, train_vgg16_layers=False):
    """Returns the discriminator model based on the VGG16 architecture pretrained on imagenet
    
    Arguments:
        input_shape {tuple(int)} -- The shape of input layer
        train_vgg16_layers {boolean} -- Whether to train VGG16 pretrained layers
    
    Returns:
        tensorflow.keras.Model -- The discriminator model
    """

    input_layer = get_input_layer(input_shape)
    vgg16 = get_vgg16_model(input_shape, input_layer)
    vgg_layers = {layer.name: layer for layer in vgg16.layers}

    block6_conv1 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(vgg_layers['block5_pool'].output)
    block6_conv2 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(block6_conv1)
    block6_conv3 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(block6_conv2)
    block6_pool = GlobalAveragePooling2D()(block6_conv3)

    fc1 = Dense(512, activation='relu')(block6_pool)
    output = Dense(1, activation='sigmoid')(block6_pool)

    model = Model(inputs=input_layer, outputs=output, name='discriminator')

    if not train_vgg16_layers:
        all_layers = {layer.name: layer for layer in model.layers}
        for name, _ in all_layers.items():
            all_layers[name].trainable = False

    return model

def get_gan_model(input_shape, generator, discriminator):
    """Returns the GAN model by combining {generator} and {discriminator} models
    
    Arguments:
        input_shape {tuple(int)} -- The shape of input layer
        generator {tensorflow.keras.Model} -- Generator model
        discriminator {tensorflow.keras.Model} -- Discriminator model
    
    Returns:
        tensorflow.keras.Model -- The GAN model
    """

    input_layer = get_input_layer(input_shape)
    generated_image = generator(input_layer)
    prediction = discriminator(generated_image)
    model = Model(inputs=input_layer, outputs=[generated_image, prediction], name='gan')

    return model
