for i in range(1):
    from os import makedirs, listdir
    import sys
    sys.path.insert(0, './code/UNET/')
    sys.path.insert(0, './code/RotNet/')
    sys.path.insert(0, './code')
    from image import ImageDataGenerator, save_img, to_categorical
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
save_masks = False
data_size = 454
BATCH_SIZE = 2
classes = 4
resize = (64,640)
n_batches = data_size/BATCH_SIZE
to_folder = '/data/predicted/mask/'+movie+'classification'
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

unet_model = unet(pretrained_weights = 'code/UNET/UNET_highW.h5', classes = classes)
predictions = unet_model.predict_generator(predict_img_generator, steps = n_batches)
final_masks = np.argmax(predictions,axis=-1)

if save_masks:
 file_names = listdir(from_path+'/frames/')
 if not os.path.exists(path+to_folder):
    os.makedirs(path+to_folder)
 for i in range(int(data_size)):
    save_img(path+to_folder+file_names[i],final_masks[i][:,:,np.newaxis]*255/classes)

new_input = to_categorical(final_masks,num_classes = classes)
objects, info = pipeline.get_objects(new_input, resize = (32,32), min_size = 60, max_size = 150)
objects.shape

for ii in range(1):
    for i in [0,1,4,5]:
        plt.figure()
        plt.scatter(np.arange(0,len(objects)),info[:,i],marker='o',c = info[:,3])
        plt.colorbar()
    plt.figure()
    plt.scatter(info[:,8],info[:,9],marker='o',c = info[:,3])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(info[:,8],info[:,9],marker='o',c = info[:,1],s = info[:,3])
    plt.colorbar()

    plt.figure()
    [plt.hist(info[info[:,1] == d,3],alpha=0.75, label=d) for d in [0,1,2,3]]
    plt.legend()

    plt.figure()
    plt.scatter(np.arange(0,len(objects)),info[:,3],marker='o',c = info[:,1])
    plt.colorbar()



plt.imshow(new_input[100,:,:,2].squeeze())


rotnet_model = rotnet(pretrained_weights = 'code/RotNet/RotNet_wAugmentation.h5', classes = 360)

# might have to break this into smaller batches
angles = rotnet_model.predict_on_batch(objects).squeeze()
angles = np.argmax(angles,axis=-1)
t = 2

import matplotlib.animation as animation
fig = plt.figure()
ims = []
for i in range(1,274):
    t=i
    a1 = plt.subplot(121)
    im1 = a1.imshow(objects[t,...].squeeze(),cmap='gray')
    a2 = plt.subplot(122)
    im2 = a2.imshow(rotate(objects[t,...],-angles[t]),cmap='gray')
    ims.append([im1,im2])

plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
ani = animation.ArtistAnimation(fig, ims, interval=500)
ani.save('code/rotations.avi')
plt.show()

for i in range(1):
    from image import array_to_img, img_to_array, load_img


    mask = load_img('code/data/predicted/mask/U/1466ul_min_1/classification/classification8871.tif', color_mode = 'grayscale',target_size=(64,640))
    mask = img_to_array(mask,dtype='uint8').squeeze()
    mask = mask

    contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    resize = (32,32)
    min_size = 50
    objects = list()
    plt.figure(figsize=(16,64))
    for k in range(len(contours[0])):
        A = cv2.contourArea(contours[0][k])
        print(A)
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

angles = rotnet_model.predict_on_batch(objects/255).squeeze()
angles = np.argmax(angles, axis= -1)
for i in range(8):
    plt.figure()
    plt.subplot(121)
    plt.imshow(objects[i,...].squeeze())
    plt.subplot(122)
    plt.imshow(rotate(objects[i,...],-angles[i]))


# TODO: save data to excel
