for i in range(1):
    import os
    import sys
    sys.path.insert(0, './code/RotNet/')
    sys.path.insert(0, './code/UNET/')
    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
    from keras.layers import Dense, Dropout, Flatten, Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Model
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from utils import angle_error_regression, RotNetDataGenerator, binarize_images, rotate
    from RotNet_model import *
    from image import ImageDataGenerator
    from scipy import ndimage

path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/AsymParticles/code/RotNet/data/stock'
nb_train_samples = 92
BATCH_SIZE = 6
nb_epoch = 500
resize = (32,32)

def noise(imgs):

    if np.random.rand(1) > 0.5:
        imgs[:,:,0] = ndimage.binary_erosion(imgs.squeeze(),np.ones((3,3))).astype(int)

    contours = cv2.findContours(imgs.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = contours[0]
    if len(cnts)>0:
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        for k in range(len(cnts)):
            x,y,w,h= boundingBoxes[k]
            if x > 0:
                x = x-1
            if y > 0:
                y = y-1
            w = w+1
            h = h+1
            obj = imgs[y:(y+h+1),x:(x+w+1)]
            imgs = cv2.resize(obj,(imgs[:,:,0].shape))[...,np.newaxis]


    if np.random.rand(1) > 0.5:
        imgs = imgs + (np.random.rand(*imgs.shape)<=0.1).astype(int)
        imgs[imgs==2] = 0

    return imgs.astype('uint8')

img_train = ImageDataGenerator(

                 shear_range = 0.2,
                 zoom_range = 0.2,
                 fill_mode = 'nearest',
                 validation_split = 0.2,
                 postprocessing_function = noise,
)
seed = 2019
train_img_generator = img_train.flow_from_directory(
                path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = 'rotate',
                subset = 'training',
                shuffle=True,
                seed = seed
)

test_img_generator = img_train.flow_from_directory(
                path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = 'rotate',
                subset = 'validation',
                shuffle=False,
                seed = seed
)

# rot_generator = RotNetDataGenerator(
#     imgs,
#     img_train,
#     input_shape = (32,32,1),
#     batch_size=BATCH_SIZE,
#     one_hot=True,
#     shuffle=True,
#     color_mode = 'grayscale',
#     preprocess_func=binarize_images
# )

callbacks = [ModelCheckpoint('code/RotNet/weights/RotNet_wNoise.h5', verbose=1, save_best_only=True, save_weights_only=True,monitor='loss'),
                ReduceLROnPlateau(monitor='loss', factor=0.75,patience=50,mdoe='min',verbose=1,min_lr=10^-6)]
rnn = rotnet(classes = 360)
h = rnn.fit_generator(
    train_img_generator,
    steps_per_epoch=nb_train_samples / BATCH_SIZE*0.8,
    epochs=nb_epoch,
    verbose=1,
    callbacks=callbacks,
    validation_data = test_img_generator,
    validation_steps = nb_train_samples*0.2/BATCH_SIZE
)

for i in range(1):
    fig = plt.figure(figsize=(8, 8));
    ax1 = fig.subplots()
    ax1.set_title("Learning curve");
    ax1.plot(h.history["loss"],'b',label="loss");
    ax1.plot(h.history["val_loss"],'b*-',label="val_loss");
    ax2 = ax1.twinx()
    ax2.plot([0],[1],'b',label="loss");
    ax2.plot(h.history["categorical_accuracy"],'r', label="cat_accuracy");
    ax2.plot(h.history["val_categorical_accuracy"],'r*-', label="val_cat_accuracy");
    ax2.plot(h.history["angle_error"],'r', label="angle_error");
    ax2.plot(h.history["val_angle_error"],'r*-', label="val_angle_error");


    # for lr in np.nonzero(np.diff(h.history['lr']))[0].tolist():
    #     ax2.plot([lr,lr],[0,1],'--',label='lr dropped')

    ax1.plot(np.argmin(h.history["loss"]), np.min(h.history["loss"]), marker="x", color="g", label="best model");
    ax1.set_xlabel("Epochs");
    ax1.set_ylabel("loss");
    plt.legend()
    plt.show()





for i in range(1):
    n_batches = 1

    i = 0
    im = np.zeros((nb_train_samples,)+(32,32)+(1,))
    an = np.zeros(nb_train_samples)
    for d, l in test_img_generator:
        im = d
        an=np.argmax(l,axis=-1)
        i += 1
        if i == n_batches:
            break

np.max(im)
im.dtype
angles = rnn.predict_on_batch(im)
angles = np.argmax(angles,axis=-1)
t = np.argmax(abs((an-angles)))

for i in range(BATCH_SIZE):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im[i,...].squeeze())
    plt.subplot(122)
    plt.imshow(rotate(im[i,...],-angles[i]))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
