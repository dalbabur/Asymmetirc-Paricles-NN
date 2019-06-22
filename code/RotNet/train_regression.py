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
    from RotNet_model import *

# we don't need the labels indicating the digit value, so we only load the images
path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/AsymParticles/code/RotNet/data/stock'
imgs = [os.path.join(path,os.path.relpath(x)) for x in os.listdir(path)]
model_name = 'rotnet_mnist_regression'

nb_train_samples = len(imgs)

# training parameters
BATCH_SIZE = 4
nb_epoch = 100

# callbacks
checkpointer = ModelCheckpoint('code/RotNet/RotNet.h5', verbose=1, save_best_only=True, save_weights_only=True,monitor='loss')
rot_generator = RotNetDataGenerator(
    imgs,
    input_shape = (32,32,1),
    batch_size=BATCH_SIZE,
    one_hot=False,
    shuffle=True,
    color_mode = 'grayscale',
    preprocess_func=binarize_images
)
# training loop
rotnet(pretrained_weights = 'code/RotNet/RotNet.h5').fit_generator(
    rot_generator,
    steps_per_epoch=nb_train_samples / BATCH_SIZE,
    epochs=nb_epoch,
    verbose=1,
    callbacks=[checkpointer]
)

for i in range(1):
    n_batches = 1

    i = 0
    im = np.zeros((nb_train_samples,)+(32,32)+(1,))
    an = np.zeros(nb_train_samples)
    for d, l in rot_generator:
        im = d
        an=l
        i += 1
        if i == n_batches:
            break

t = 3
for i in range(1):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im[t,...].squeeze())
    plt.subplot(122)
    plt.imshow(rotate(im[t,...],-an[t]*360))
