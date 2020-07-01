import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from neural_net_training.batch_generator import BatchDataGen
from neural_net_training.resnet import SSIM, unet
from rand_functions.helper_functions import gen_file_list

train_data_path = "./train_data/reconstructed"
train_mask_path = "./train_data/ground_truth"
valid_data_path = "./valid_data/reconstructed"
valid_mask_path = "./valid_data/ground_truth"

num_files_per_nrrd = 128
batch_size = 4

train_steps = len(gen_file_list(train_data_path)) * num_files_per_nrrd // batch_size
valid_steps = len(gen_file_list(valid_data_path)) * num_files_per_nrrd // batch_size
x_batch, y_batch = next(BatchDataGen(train_data_path, train_mask_path, batch_size).gen_batch())
train_datagen = BatchDataGen(train_data_path, train_mask_path, batch_size).gen_batch()
valid_datagen = BatchDataGen(valid_data_path, valid_mask_path, batch_size).gen_batch()

_callbacks = [
    ModelCheckpoint(filepath='resnet_{epoch:02d}.h5', monitor='val_loss', mode='min', save_best_only=True,
                    save_freq=10),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
]

model = unet(2, x_batch.shape[1:], filters=[32, 64, 128, 256, 512], actfn='relu', is_batchnorm=True,
             lastActivation=None, residual=True)

opt = tf.keras.optimizers.Adam(1e-5)
model.summary()
model.compile(optimizer=opt, loss=SSIM)

trained_model = model.fit(train_datagen, steps_per_epoch=train_steps, epochs=25,
                          validation_data=valid_datagen, validation_steps=valid_steps,
                          callbacks=_callbacks)

# model = tf.keras.models.load_model("resnet_10.h5", custom_objects={'loss': SSIM}, compile=False)
# X, Y = next(BatchDataGen(valid_data_path, valid_mask_path, 4).gen_batch())
# y_pred = model.predict(x_pred)
# y_pred = np.squeeze(pred)
# Y = np.squeeze(Y)
# X = np.squeeze(X)
# nrrd.write('resnet_pred.nrrd', y_pred, compression_level=1, index_order='C')
# nrrd.write('resnet_Y.nrrd', Y, compression_level=1, index_order='C')
# nrrd.write('resnet_X.nrrd', X, compression_level=1, index_order='C')
