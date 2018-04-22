import tensorflow as tf

VGG16 = tf.keras.applications.vgg16.VGG16
Input = tf.keras.layers.Input
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
UpSampling2D = tf.keras.layers.UpSampling2D
Add = tf.keras.layers.Add
Model = tf.keras.models.Model


def get_vgg16_model(input_shape, input_layer):
    return VGG16(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=input_layer)


def get_generator_model():
    input_shape = (224, 224, 3)
    input_layer = Input(shape=input_shape)
    model = get_vgg16_model(input_shape, input_layer)
    layers = {layer.name: layer for layer in model.layers}

    block5_bn = BatchNormalization()(layers['block5_pool'].output)
    block5_bn_conv = Conv2D(filters=512, kernel_size=1, activation='relu', padding='same')(block5_bn)
    block5_up = UpSampling2D(size=(2, 2))(block5_bn_conv)

    block4_bn = BatchNormalization()(layers['block4_pool'].output)
    block45_add = Add()([block4_bn, block5_up])
    block45_add_conv = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(block45_add)
    block45_up = UpSampling2D(size=(2, 2))(block45_add_conv)

    block3_bn = BatchNormalization()(layers['block3_pool'].output)
    block34_add = Add()([block3_bn, block45_up])
    block34_add_conv = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(block34_add)
    block34_up = UpSampling2D(size=(2, 2))(block34_add_conv)

    block2_bn = BatchNormalization()(layers['block2_pool'].output)
    block23_add = Add()([block2_bn, block34_up])
    block23_add_conv = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(block23_add)
    block23_up = UpSampling2D(size=(2, 2))(block23_add_conv)

    block1_bn = BatchNormalization()(layers['block1_pool'].output)
    block12_add = Add()([block1_bn, block23_up])
    block12_add_conv = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(block12_add)

    block12_up = UpSampling2D(size=(2, 2))(block12_add_conv)
    output = Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='same')(block12_up)

    return Model(inputs=input_layer, outputs=output)
