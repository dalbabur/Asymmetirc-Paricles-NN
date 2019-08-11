for i in range(1):
    import os
    import sys
    sys.path.insert(0, './code/ClassNet/')
    sys.path.insert(0, './code/UNET/')
    sys.path.insert(0, './code/')
    from keras.callbacks import *
    from keras.layers import Dense, Dropout, Flatten, Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Model
    import numpy as np
    import matplotlib.pyplot as plt
    from ClassNet_model import *
    from image import ImageDataGenerator, generate_rotated_image
    from keract import *
    from keras.backend import categorical_crossentropy as cc
    from keras.backend import variable, eval
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report as class_report
    from keract import *
    from scipy import ndimage
    import cv2
    import data_vis_tools as dv

path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/AsymParticles/code/UNET/data/synthetic/particles/masks/'
data_size = 150
BATCH_SIZE = 20
nb_epoch = 200
resize = (32,32)
classes = 3
weights = 'code/ClassNet/weights/ClassNet4slim.h5'

def no_noise(img):
    imgs = img.copy()
    e = np.random.randint(1,4)
    imgs[:,:,0] = ndimage.binary_erosion(imgs.squeeze(),np.ones((e,e))).astype(int)


    if np.random.rand(1) > 0.05:
        rotation_angle = np.random.randint(360)
        imgs = generate_rotated_image(
            imgs,
            rotation_angle,
            size=imgs.shape[:2]
        )
        if imgs.ndim == 2:
            imgs = np.expand_dims(imgs, axis=2)

    if np.random.rand(1) > 0.5:
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

                if np.sum(imgs) > 32*32*0.7:
                    imgs = img.copy()

    imgs = (imgs > 0.5).astype('uint8')
    return imgs

def noise(img):
    imgs = img.copy()
    e = np.random.randint(1,4)
    imgs[:,:,0] = ndimage.binary_erosion(imgs.squeeze(),np.ones((e,e))).astype(int)


    if np.random.rand(1) > 0.05:
        rotation_angle = np.random.randint(360)
        imgs = generate_rotated_image(
            imgs,
            rotation_angle,
            size=imgs.shape[:2]
        )
        if imgs.ndim == 2:
            imgs = np.expand_dims(imgs, axis=2)

    if np.random.rand(1) > 0.5:
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

                if np.sum(imgs) > 32*32*0.7:
                    imgs = img.copy()

    if np.random.rand(1) > 0.5:
        imgs = imgs + (np.random.rand(*imgs.shape)<=np.random.uniform(0.075,0.13,1)).astype(int)
        imgs[imgs>=2] = 0

    imgs = (imgs > 0.5).astype('uint8')
    return imgs

img_train = ImageDataGenerator(
                 preprocessing_function = noise)

img_test = ImageDataGenerator(
                 preprocessing_function = no_noise
)

seed = 1997
train_img_generator = img_train.flow_from_directory(
                path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = 'categorical_bin',
                seed = seed
)

test_img_generator = img_test.flow_from_directory(
                path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = 'categorical_bin',
                seed = seed
)

cnn = classnet(classes = classes, input_size = (*resize,1))

callbacks = [ModelCheckpoint(weights, verbose=1, save_best_only=True, save_weights_only=True,monitor='loss'),
                ReduceLROnPlateau(monitor='loss', factor=0.75,patience=15,mode='min',verbose=1,min_lr=10^-6)]

h = cnn.fit_generator(
    train_img_generator,
    steps_per_epoch = data_size/BATCH_SIZE,
    epochs=nb_epoch,
    verbose=1,
    callbacks=callbacks,
    validation_data = test_img_generator,
    validation_steps = data_size/BATCH_SIZE,
)

for i in range(1):
    fig = plt.figure(figsize=(8, 8));
    ax1 = fig.subplots()
    ax1.set_title("Learning curve");
    ax1.plot(h.history["loss"],'b',label="loss");
    ax1.plot(h.history["val_loss"],'b--',label="val_loss");
    ax2 = ax1.twinx()
    ax2.plot([0],[1],'b',label="loss");
    ax2.plot(h.history["categorical_accuracy"],'r', label="cat_accuracy");
    ax2.plot(h.history["val_categorical_accuracy"],'r--', label="val_cat_accuracy");

    for lr in np.nonzero(np.diff(h.history['lr']))[0].tolist():
        ax2.plot([lr,lr],[0,1],'--',label='lr dropped')

    ax1.plot(np.argmin(h.history["loss"]), np.min(h.history["loss"]), marker="x", color="g", label="best model");
    ax1.set_xlabel("Epochs");
    ax1.set_ylabel("loss");
    ax2.set_ylabel("accuracy")
    plt.legend()
    plt.show()

for i in range(1):
    n = 500
    i = 0
    test_imgs,test_labels = list(),list()
    for imgs,labels in ImageDataGenerator(rotation_range = 360).flow_from_directory(
                    path,
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = 1,
                    class_mode = 'categorical_bin',
                    seed = seed
    ):
        test_imgs.append(imgs)
        test_labels.append(labels)
        i += 1
        if i == n:
            break
    test_imgs = np.reshape(np.array(test_imgs),(-1,*resize,1))
    test_labels = np.reshape(np.array(test_labels),(-1,classes))

    predictions = cnn.predict_on_batch(test_imgs)
    dv.plot_confusion_matrix(np.argmax(test_labels,1),np.argmax(predictions,1),classes=['L','U','UFO'],normalize = True)

    plt.figure()
    plt.hist(np.argmax(test_labels,1),alpha = 0.5)
    plt.hist(np.argmax(predictions,1),alpha = 0.5)

    missed = test_imgs[(np.argmax(predictions,1)-np.argmax(test_labels,1)) != 0,...]
    missed_labels = np.argmax(predictions,1)[(np.argmax(predictions,1)-np.argmax(test_labels,1)) != 0]

    print(class_report(np.argmax(test_labels,1),np.argmax(predictions,1)))

for i in range(len(missed)):
    plt.figure()
    plt.imshow(missed[i,...].squeeze())
    plt.title(missed_labels[i])

for i in range(20):
    plt.figure()
    plt.subplot(121)
    plt.imshow(test_imgs[i,...].squeeze())
    # plt.subplot(122)
    # plt.imshow(noise(test_imgs[i,...]).squeeze())
