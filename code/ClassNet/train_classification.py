for i in range(1):
    import os
    import sys
    sys.path.insert(0, './code/ClassNet/')
    sys.path.insert(0, './code/UNET/')
    from keras.callbacks import *
    from keras.layers import Dense, Dropout, Flatten, Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Model
    import numpy as np
    import matplotlib.pyplot as plt
    from ClassNet_model import *
    from image import ImageDataGenerator
    from keract import *
    from keras.backend import categorical_crossentropy as cc
    from keras.backend import variable, eval
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report as class_report
    from keract import *
    from scipy import ndimage
    import cv2

path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/AsymParticles/code/UNET/data/synthetic/particles/masks'

data_size = 150
BATCH_SIZE = 20
nb_epoch = 400
resize = (32,32)
classes = 3

def noise(imgs):

    if max(imgs.flat) != 1:
        imgs[imgs>0] = 1

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
        imgs = imgs + (np.random.rand(*imgs.shape)<=0.13).astype(int)
        imgs[imgs==2] = 0

    return imgs.astype(int)

img_train = ImageDataGenerator(

                 fill_mode = 'constant',
                 cval = 0,
                 rotation_range = 360,
                 validation_split = 0.2,
                 preprocessing_function = noise,
)

seed = 2019
train_img_generator = img_train.flow_from_directory(
                path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = 'categorical_bin',
                subset = 'training',
                seed = seed
)

test_img_generator = img_train.flow_from_directory(
                path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = 'categorical_bin',
                subset = 'validation',
                seed = seed
)

cnn = classnet(classes = classes, input_size = (*resize,1))
cnn.summary()

callbacks = [ModelCheckpoint('code/ClassNet/ClassNet2.h5', verbose=1, save_best_only=True, save_weights_only=True,monitor='loss'),
                ReduceLROnPlateau(monitor='loss', factor=0.75,patience=15,mdoe='min',verbose=1,min_lr=10^-6)]

h = cnn.fit_generator(
    train_img_generator,
    steps_per_epoch = data_size*0.8/BATCH_SIZE,
    epochs=nb_epoch,
    verbose=1,
    callbacks=callbacks,
    validation_data = test_img_generator,
    validation_steps = data_size*0.2/BATCH_SIZE,
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

    for lr in np.nonzero(np.diff(h.history['lr']))[0].tolist():
        ax2.plot([lr,lr],[0,1],'--',label='lr dropped')

    ax1.plot(h.history["tversky"],'g',label="loss");
    ax1.plot(h.history["val_tversky"],'g*-',label="val_loss");

    ax1.plot(h.history["dice_coef_loss"],'y',label="loss");
    ax1.plot(h.history["val_dice_coef_loss"],'y*-',label="val_loss");

    ax1.plot(h.history["lossFunc"],'k',label="loss");
    ax1.plot(h.history["val_lossFunc"],'k*-',label="val_loss");

    ax1.plot(np.argmin(h.history["loss"]), np.min(h.history["loss"]), marker="x", color="g", label="best model");
    ax1.set_xlabel("Epochs");
    ax1.set_ylabel("loss");
    plt.legend()
    plt.show()

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

cnn = classnet(pretrained_weights = 'code/ClassNet/ClassNet3.h5',classes = classes, input_size = (*resize,1))
for i in range(1):
    n = 1
    i = 0
    test_imgs,test_labels = list(),list()
    for imgs,labels in ImageDataGenerator().flow_from_directory(
                    path,
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = 150,
                    class_mode = 'categorical_bin',
    ):
        test_imgs.append(imgs)
        test_labels.append(labels)
        i += 1
        if i == n:
            break
    test_imgs = np.reshape(np.array(test_imgs),(-1,*resize,1))
    test_labels = np.reshape(np.array(test_labels),(-1,classes))


    predictions = cnn.predict_on_batch(test_imgs)
    plot_confusion_matrix(np.argmax(test_labels,1),np.argmax(predictions,1),classes=['L','U','UFO'],normalize = True)
    class_report(np.argmax(test_labels,1),np.argmax(predictions,1))

    plt.figure()
    plt.hist(np.argmax(test_labels,1),alpha = 0.5)
    plt.hist(np.argmax(predictions,1),alpha = 0.5)

    plt.figure()
    plt.imshow(test_imgs[-8,...].squeeze())

    missed = test_imgs[(np.argmax(predictions,1)-np.argmax(test_labels,1)) != 0,...]

    for i in range(len(missed[:,0,0,0])):
        plt.figure()
        plt.imshow(missed[i,...].squeeze())

np.argmax(cnn.predict(test_imgs[52,...][np.newaxis]))
for i in range(len(test_imgs[:,0,0,0])):
    plt.figure()
    plt.imshow(test_imgs[i,...].squeeze())

act = get_activations(cnn,test_imgs[0,...][np.newaxis])
display_activations(act)

testing = test_imgs[50,...].squeeze()
e1 = np.ones((test_imgs.shape))
(ndimage.binary_erosion(test_imgs,e1).astype(int))
e1.shape
max(test_imgs[:].flat)
plt.imshow(cv2.resize(test_imgs[4,...],(28,28)))
