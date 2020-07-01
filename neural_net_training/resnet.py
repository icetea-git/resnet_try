import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D, UpSampling2D, UpSampling3D
from tensorflow.keras.layers import Input, Conv2D, Conv3D, BatchNormalization, Concatenate
from tensorflow.keras.models import Model


def UnetConv(dim, input, filtersnum, is_batchnorm, actfn, name, res=True):
    block = tf.keras.models.Sequential(name=name)

    block.add(ConvSolution(dim, filtersnum, kernel_initializer="glorot_normal", padding="same", name=name + '_1'))
    if is_batchnorm:
        block.add(BatchNormalization(name=name + '_1_bn'))
        block.add(Activation(actfn, name=name + '_1_act'))
        block.add(ConvSolution(dim, filtersnum, kernel_initializer="glorot_normal", padding="same", name=name + '_2'))
    if is_batchnorm:
        block.add(BatchNormalization(name=name + '_2_bn'))
        block.add(Activation(actfn, name=name + '_2_act'))
    if (res):
        if (dim == 2):
            projected_input = Conv2D(filtersnum, (1, 1), strides=(1, 1), kernel_initializer="glorot_normal",
                                     padding="same", name=name + '_proj')(input)
        elif (dim == 3):
            projected_input = Conv3D(filtersnum, (1, 1, 1), strides=(1, 1, 1), kernel_initializer="glorot_normal",
                                     padding="same", name=name + '_proj')(input)
        x = block(input) + projected_input
        x = Activation(actfn, name=name + '_res_act')(x)
    else:
        x = block(input)

    return x


def ConvSolution(dim, filtersnum, kernel_initializer, padding, name):
    if (dim == 2):
        return Conv2D(filtersnum, (3, 3), strides=(1, 1), kernel_initializer=kernel_initializer, padding=padding,
                      name=name)
    elif (dim == 3):
        return Conv3D(filtersnum, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kernel_initializer, padding=padding,
                      name=name)


def PoolingSolution(dim):
    if (dim == 2):
        return AveragePooling2D(pool_size=(2, 2))
    elif (dim == 3):
        return AveragePooling3D(pool_size=(2, 2, 2))


def UpsamplingSolution(dim):
    if (dim == 2):
        return UpSampling2D()
    elif (dim == 3):
        return UpSampling3D()


def unet(dim, input_size, filters, actfn, is_batchnorm, lastActivation=None, residual=False):
    '''
    Direct 3D analogue of the above defined unet.
    '''
    m = {}
    m["inputs"] = Input(shape=input_size)

    last = m["inputs"]
    for d in range(len(filters) - 1):
        m[f"conv{d}_l"] = UnetConv(dim, last, filters[d], is_batchnorm, actfn, name=f"conv{d}_l", res=residual)
        m[f"pool{d}"] = PoolingSolution(dim)(m[f"conv{d}_l"])
        last = m[f"pool{d}"]
        d += 1
        m[f"conv{d}_r"] = UnetConv(dim, last, filters[d], is_batchnorm, actfn, name=f"conv{d}_r", res=residual)

    for d in range(len(filters) - 1, 0, -1):
        m[f"up{d - 1}_r"] = Concatenate(axis=-1)([UpsamplingSolution(dim)(m[f"conv{d}_r"]), m[f"conv{d - 1}_l"]])
        m[f"conv{d - 1}_r"] = UnetConv(dim, m[f"up{d - 1}_r"], filters[d - 1], is_batchnorm, actfn,
                                       name=f"conv{d - 1}_r",
                                       res=residual)
    if (dim == 2):
        m[f"conv{0}_r"] = Conv2D(1, (1, 1), activation=lastActivation, name='final', dtype='float32')(
            m[f"conv{d - 1}_r"])  # mixed precision training
    elif (dim == 3):
        m[f"conv{0}_r"] = Conv3D(1, (1, 1, 1), activation=lastActivation, name='final', dtype='float32')(
            m[f"conv{d - 1}_r"])  # mixed precision training

    model = Model(inputs=[m["inputs"]], outputs=[m[f"conv{0}_r"]])
    return model

def SSIM(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))