for i in range(1):
    import os
    import sys
    sys.path.insert(0, './code/RotNet/')
    from keras.callbacks import ModelCheckpoint
    from keras.layers import Dense, Dropout, Flatten, Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Model
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import angle_error_regression, RotNetDataGenerator, binarize_images, rotate

# we don't need the labels indicating the digit value, so we only load the images
path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/AsymParticles/code/RotNet/data/stock'
imgs = [os.path.join(path,os.path.relpath(x)) for x in os.listdir(path)]
model_name = 'rotnet_mnist_regression'

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

nb_train_samples, img_rows, img_cols = (len(imgs),43,43)
img_channels = 1
input_shape = (img_rows, img_cols, img_channels)

# model definition
input = Input(shape=(img_rows, img_cols, img_channels))
x = Conv2D(nb_filters, kernel_size, activation='relu')(input)
x = Conv2D(nb_filters, kernel_size, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=x)

# model compilation
model.compile(loss=angle_error_regression,
              optimizer='adam')

# training parameters
BATCH_SIZE = 4
nb_epoch = 50

# callbacks
checkpointer = ModelCheckpoint('code/RotNet/RotNet.h5', verbose=1, save_best_only=True, save_weights_only=True,monitor='loss')
rot_generator = RotNetDataGenerator(
    imgs,
    input_shape = (43,43,1),
    batch_size=BATCH_SIZE,
    one_hot=False,
    shuffle=True,
    color_mode = 'grayscale',
    preprocess_func=binarize_images
)
# training loop
model.fit_generator(
    rot_generator,
    steps_per_epoch=nb_train_samples / BATCH_SIZE,
    epochs=nb_epoch,
    verbose=1,
    callbacks=[checkpointer]
)

for i in range(1):
    n_batches = 1

    i = 0
    im = np.zeros((nb_train_samples,)+(43,43)+(1,))
    an = np.zeros(nb_train_samples)
    for d, l in rot_generator:
        im = d
        an=l
        i += 1
        if i == n_batches:
            break

t = 2
for i in range(1):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im[t,...].squeeze())
    plt.subplot(122)
    plt.imshow(rotate(im[t,...],-an[t]*360))
