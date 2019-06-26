for i in range(1):
    from os import makedirs, listdir
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

movie = '/L/1466ul_min_2/'
from_path = 'code/data/authentic'+movie
save_masks = True
data_size = 454
BATCH_SIZE = 4
classes = 4
resize = (64,640)
n_batches = data_size/BATCH_SIZE
to_folder = '/data/predicted/mask/'+movie
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

unet_model = unet(pretrained_weights = 'code/UNET/UNET.h5', classes =classes)
predictions = unet_model.predict_generator(predict_img_generator, steps = n_batches)

if save_masks:
 final_masks = np.argmax(predictions,axis=-1)
 file_names = listdir(from_path+'/frames/')
 if not os.path.exists(path+to_folder):
    os.makedirs(path+to_folder)
 for i in range(int(data_size)):
    save_img(path+to_folder+file_names[i],final_masks[i][:,:,np.newaxis]*255/classes)

objects, info = pipeline.get_objects(predictions*255, resize = (32,32), min_size = 100)


# info[np.argsort(info[:,3])][70]
#
# np.argmax(np.sort(info[:,3]) > 180)
# plt.imshow(objects[np.argsort(info[:,3])][695,...].squeeze())
# plt.imshow(p[2020,:,:,1].squeeze())


rotnet_model = rotnet(pretrained_weights = 'code/RotNet/RotNet2.h5')

# might have to break this into smaller batches
angles = rotnet_model.predict_on_batch(objects/255).squeeze()

t = 2
for i in range(7):
    t=i
    plt.figure()
    plt.subplot(121)
    plt.imshow(objects[t,...].squeeze(),cmap='gray')
    plt.subplot(122)
    plt.imshow(rotate(objects[t,...],-angles[t]*360),cmap='gray')


for i in range(1):
    from image import array_to_img, img_to_array, load_img


    mask = load_img('code/data/predicted/mask/3626_scaled.tif', color_mode = 'grayscale',target_size=(64,640))
    mask = img_to_array(mask,dtype='uint8').squeeze()
    mask = mask

    contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    resize = (32,32)
    min_size = 50
    objects = list()
    plt.figure(figsize=(16,64))
    for k in range(len(contours[0])):
        A = cv2.contourArea(contours[0][k])
        if A > min_size:
            x,y,w,h = cv2.boundingRect((contours[0][k]))
            if x > 0:
                x = x-1
            if y > 0:
                y = y-1
            w = w+1
            h = h+1
            plt.imshow(cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),2), cmap='gray')
            obj = mask[y:(y+h+1),x:(x+w+1)]
            if resize is not None:
                obj = cv2.resize(obj,resize)
            objects.append(obj)
    if resize is not None:
        objects = np.array(objects)[:,:,:,np.newaxis]
