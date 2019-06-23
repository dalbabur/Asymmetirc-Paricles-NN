for i in range(1):
    import sys
    sys.path.insert(0, './code/UNET/')
    sys.path.insert(0, './code/RotNet/')
    sys.path.insert(0, './code')
    from image import ImageDataGenerator, save_img
    from utils import RotNetDataGenerator
    import matplotlib.pyplot as plt
    import cv2
    from UNET_model import *
    from RotNet_model import *
    import pipeline
    from keras.callbacks import *
    import numpy as np
    from keract import *
    from importlib import reload
    pipeline = reload(pipeline)

from_path = 'code/data/authentic/U/'
data_size = 4137
BATCH_SIZE = 7
classes = 4
resize = (64,640)
n_batches = data_size/BATCH_SIZE
to_folder = '/data/predicted/mask/'
path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/AsymParticles/code'

predict_gen = ImageDataGenerator(rescale = 1./255)
predict_img_generator = predict_gen.flow_from_directory(
                from_path,
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = None,
                shuffle = False
)

unet = unet(pretrained_weights = 'code/UNET/UNET.h5', classes =classes)
predictions = unet.predict_generator(predict_img_generator, steps = n_batches)

# final_masks = np.argmax(predictions,axis=-1)
# for i in range(int(data_size)):
#     save_img(path+to_folder+str(i)+'_scaled.tif',final_masks[i][:,:,np.newaxis]*255/classes)

objects, info = pipeline.get_objects(predictions*255, resize = (32,32), min_size = 100)


# info[np.argsort(info[:,3])][70]
#
# np.argmax(np.sort(info[:,3]) > 180)
# plt.imshow(objects[np.argsort(info[:,3])][695,...].squeeze())
# plt.imshow(p[2020,:,:,1].squeeze())


rotnet = rotnet(pretrained_weights = 'code/RotNet/RotNet.h5')

# might have to break this into smaller batches
angles = rotnet.predict_on_batch(objects).squeeze()

t = 100
for i in range(1):
    plt.figure()
    plt.subplot(121)
    plt.imshow(objects[t,...].squeeze())
    plt.subplot(122)
    plt.imshow(rotate(objects[t,...],-angles[t]*360))
