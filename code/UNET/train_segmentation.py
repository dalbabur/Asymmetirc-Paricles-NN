for i in range(1):
    import sys
    sys.path.insert(0, './code/UNET/')
    from image import ImageDataGenerator, array_to_img, img_to_array, load_img
    import matplotlib.pyplot as plt
    import cv2
    from model import *
    from keras.callbacks import *
    import numpy as np
    from keract import *

data_size = 2560
test_size = 512
classes = 4
BATCH_SIZE = 8
resize = (64,640)

for i in range(1):
    img_train = ImageDataGenerator(                             # TODO: FIGURE OUT TRANSFORMATION ON MASK (rn it spills to other classes)

                    rescale = 1./255,
                    # rotation_range = 1,
                    # width_shift_range = 0.02,
                    # height_shift_range = 0.02,
                    # brightness_range = (0.5,1.5),
                    # shear_range = 0.1,
                    # zoom_range = 0.1,
                    # horizontal_flip = True,
                    # fill_mode = 'nearest'
    )

    mask_train = ImageDataGenerator(

                    # rotation_range = 1,
                    # width_shift_range = 0.02,
                    # height_shift_range = 0.02,
                    # shear_range = 0.1,
                    # zoom_range = 0.1,
                    # horizontal_flip = True,
                    # fill_mode = 'constant',
                    # cval = 0

    )

    img_test = ImageDataGenerator(rescale = 1./255)
    mask_test = ImageDataGenerator()

    seed = 2019
    train_img_generator = img_train.flow_from_directory(
                    'data/full/img',
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = BATCH_SIZE,
                    #save_to_dir = 'data/preview/img',
                    class_mode = None,
                    seed = seed
    )

    train_mask_generator = mask_train.flow_from_directory(
                    'data/full/mask',
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = BATCH_SIZE,
                    #save_to_dir = 'data/preview/mask',
                    class_mode = 'HOT',
                    total_classes = classes,
                    seed = seed
    )

    test_img_generator = img_test.flow_from_directory(
                    'data/test/img',
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = BATCH_SIZE,
                    class_mode = None,
                    seed = seed
    )

    test_mask_generator = mask_test.flow_from_directory(
                    'data/test/mask',
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = BATCH_SIZE,
                    class_mode = 'HOT',
                    total_classes = classes,
                    seed = seed
    )

    train_generator = zip(train_img_generator, train_mask_generator)
    test_generator = zip(test_img_generator, test_mask_generator)
    callbacks = [                                                 # TODO: MONITOR VAL_LOSS IF POSSIBLE
        #EarlyStopping(patience=5, verbose=1, monitor = 'loss'),    # there seems to be some problem with ES and RLROP, possibly caused by PATIENCE
        ModelCheckpoint('testFULL.h5', verbose=1, save_best_only=True, save_weights_only=True, monitor = 'loss'),
        ReduceLROnPlateau(monitor='loss', factor=0.2,patience=3,mdoe='min',verbose=1,cooldown=1)
    ]

h = unet(classes = classes).fit_generator(                          # TODO: FIGURE OUT IF USING VALIDATOIN IS POSSIBLE (rn bathces become all crazy)
                train_generator,
                epochs = 10,                                        # remmeber you can continue training if you just load weights
                steps_per_epoch = data_size/BATCH_SIZE,
                #validation_data= test_generator,
                #validation_steps=12,
                callbacks = callbacks                               # TODO: FIGURE OUT IF CLASS_WEIGHT WORKS
)

for i in range(1):
    fig = plt.figure(figsize=(8, 8));
    ax1 = fig.subplots()
    ax1.set_title("Learning curve");
    ax1.plot(h.history["loss"],'b',label="loss");
    ax2 = ax1.twinx()
    ax2.plot([0],[1],'b',label="loss");
    ax2.plot(h.history["categorical_accuracy"],'r', label="cat_accuracy");
    for lr in np.nonzero(np.diff(h.history['lr']))[0].tolist():
        ax2.plot([lr,lr],[0.9,1],'--',label='lr dropped')
    ax1.plot(np.argmin(h.history["loss"]), np.min(h.history["loss"]), marker="x", color="r", label="best model");
    ax1.set_xlabel("Epochs");
    ax1.set_ylabel("log_loss");
    plt.legend()
    plt.show()

model = unet(pretrained_weights = 'testFULL.h5', classes =classes)
model.evaluate_generator(test_generator, steps=test_size/BATCH_SIZE)

for i in range(1):
    n_batches = 1

    i = 0
    imgs = np.zeros((n_batches*BATCH_SIZE,)+resize+(1,))
    masks = np.zeros((n_batches*BATCH_SIZE,)+resize+(4,))
    for d, l in train_generator:
        imgs[i*BATCH_SIZE:((BATCH_SIZE)+i*BATCH_SIZE),...] = d
        masks[i*BATCH_SIZE:((BATCH_SIZE)+i*BATCH_SIZE),...] = l
        i += 1
        if i == n_batches:
            break

    prediction = np.zeros(masks.shape)
    for k in range(n_batches*BATCH_SIZE):
        prediction[k] = list(get_activations(model,imgs[k][np.newaxis],'conv2d_46').values())[0]

    for i in range(n_batches*BATCH_SIZE):
        fig = plt.figure(figsize=(16, 64))
        gs = fig.add_gridspec(2,6)

        ax = fig.add_subplot(gs[0:2, 0:2])
        ax.imshow(imgs[i].squeeze())

        ax = fig.add_subplot(gs[0,2])
        ax.imshow(prediction[i,:,:,0])
        ax = fig.add_subplot(gs[0,3])
        ax.imshow(prediction[i,:,:,1])
        ax = fig.add_subplot(gs[1,2])
        ax.imshow(prediction[i,:,:,2])
        ax = fig.add_subplot(gs[1,3])
        ax.imshow(prediction[i,:,:,3])

        ax = fig.add_subplot(gs[0,4])
        ax.imshow(masks[i,:,:,0])
        ax = fig.add_subplot(gs[0,5])
        ax.imshow(masks[i,:,:,1])
        ax = fig.add_subplot(gs[1,4])
        ax.imshow(masks[i,:,:,2])
        ax = fig.add_subplot(gs[1,5])
        ax.imshow(masks[i,:,:,3])
