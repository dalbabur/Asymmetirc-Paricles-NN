for i in range(1):
    from os import makedirs, listdir
    import sys
    sys.path.insert(0, './code/UNET/')
    sys.path.insert(0, './code/RotNet/')
    sys.path.insert(0, './code/ClassNet/')
    sys.path.insert(0, './code')
    from image import ImageDataGenerator, save_img, to_categorical, load_img, img_to_array
    from utils import RotNetDataGenerator, rotate
    import matplotlib.pyplot as plt
    import cv2
    from UNET_model import *
    from RotNet_model import *
    from ClassNet_model import *
    import pipeline2
    from keras.callbacks import *
    import numpy as np
    from keract import *
    from importlib import reload
    pipeline2 = reload(pipeline2)

movie = '/L/1466ul_min_2/'
from_path = 'code/data/authentic'+movie
save_masks = False
data_size = 454
BATCH_SIZE = 2
classes = 2
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

unet_model = unet(pretrained_weights = 'code/UNET/UNET_bin.h5', classes = classes)
predictions = unet_model.predict_generator(predict_img_generator, steps = n_batches)
final_masks = np.argmax(predictions,axis=-1)
new_input = to_categorical(final_masks,num_classes = classes)
new_input2 = (final_masks>0).astype(int);
predictions.shape
new_input2.shape

for i in range(1):

    i = 0
    for d in predict_gen.flow_from_directory(
                    from_path,
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = data_size,
                    class_mode = None,
                    shuffle = False
    ):
        i += 1
        if i == n_batches:
            break
    start = 0
    n_batches = 20
    for t in range(start,start+n_batches*BATCH_SIZE):
        # plt.figure(figsize=(16, 64))
        # plt.imshow(d[t,...].squeeze())
        # plt.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r*')
        # plt.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g*')
        plt.figure(figsize=(16, 64))
        plt.imshow(final_masks[t,...])
        plt.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r*')
        plt.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g*')
        plt.title(t)

import matplotlib.animation as animation
fig = plt.figure(figsize=(64,14))
ims = []
for i in range(len(d[:,0,0,0])):
    t=i
    a1 = plt.subplot(111)
    coms, = a1.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r.',markersize=30)
    boxs, = a1.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g.',markersize=30)
    im1 = a1.imshow(d[t,...].squeeze(),cmap='gray')
    ims.append([im1,coms,boxs])

plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
ani = animation.ArtistAnimation(fig, ims, interval=300)
ani.save('code/masks.avi')
plt.show()

if save_masks:
 file_names = listdir(from_path+'/frames/')
 if not os.path.exists(path+to_folder):
    os.makedirs(path+to_folder)
 for i in range(int(data_size)):
    save_img(path+to_folder+file_names[i],final_masks[i][:,:,np.newaxis])

cnn = classnet(classes = 3 ,pretrained_weights = 'code/ClassNet/ClassNet2.h5',input_size = (32,32,1))
objects, info = pipeline2.get_objects(new_input2, cnn, resize = (32,32), min_size = 0, max_size = 1600)
objects.shape
info.shape
np.unique(info[:,0]).shape

# [0 1 2  3 4 5 6 7 8   9   10 ]
# [i,k,c,pc,A,x,y,w,h,com1,com2]

for ii in range(1):
    plt.figure(figsize = (8,6))
    plt.scatter(info[:,2], info[:,3], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(np.arange(len(objects)), info[:,9], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(info[:,5], info[:,6], marker='o-', c = info[:,0], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(info[:,9], info[:,10], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.plot(info[:,0],info[:,6],'o')
    plt.colorbar()

    plt.figure(figsize = (8,6))
    [plt.hist(info[info[:,2] == d,4], alpha=0.75, label=d) for d in [0,1,2]]
    plt.legend()

for i in range(50):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(objects[i,:,:].squeeze())
    plt.title([info[i,2],info[i,3]])
    plt.subplot(1,2,2)
    plt.imshow(rotate(objects[i,...].squeeze(),-angles[i]))
    plt.title([-angles[i],info[i,9],info[i,10]])

traj = []
dists = np.ones((565,565))*9999
k = np.zeros(565)
memory = []
i = 0
idx = 0
labels = []
while i  < info.shape[0]:
    idx = i
    dummy = list(labels)
    for f in range(sum(info[:,0] == i)):
        idx = i+f
        print(['idx',idx,'i',i,'f',f,'total',sum(info[:,0] == i)])
        if memory == []:
            traj.append(info[idx,:])
            if i == 0:
                labels.append(0)
            else:
                labels.append(max(labels)+1)
            print(['empty memory, new object',labels[-1]])
        else:
            dists = np.zeros(len(memory))
            for j in range(len(memory)):
                dists[j] = ( (info[idx,9] - memory[j][9])**2 + (info[idx,10] - memory[j][10])**2 + (info[idx,0] - memory[j][0])**2)**(1/2)

            print(dists)
            print(['argmin',np.argmin(dists)])
            print([dummy,dummy[-len(memory):]])
            label = dummy[-len(memory):][np.argmin(dists)]
            if min(dists) < 15:
                if k[label] == 0:
                    traj[label] = np.append([traj[label]],[info[idx,:]],axis=0)
                    labels.append(label)
                    print(['first append to old object', label])
                else:
                    traj[label] = np.append(traj[label],[info[idx,:]],axis=0)
                    print(['append to old object',label])
                    labels.append(label)
                k[label] = k[label]+1
            else:
                traj.append(info[idx,:])
                labels.append(max(labels)+1)
                print(['too far, new object',labels[-1]])

    memory = []
    print(['cleared memory'])
    for f in range(sum(info[:,0] == i)):
        memory.append(info[i+f,:])
        print(['added memory'])
    print(['length memory',len(memory)])
    i = idx+1
    print('-----------------------')

plt.plot(k)
len(traj[3])
info[0:20,0]
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

    info = list()
    objects = list()
    resize = (32,32)
    min_size = 0
    max_size = 1600
    i = 20
    bin = new_input2[i,:,:][:,:,np.newaxis].astype('uint8')
    contours = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = contours[0]
    if len(cnts)>0:
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][1], reverse=False))

        for k in range(len(cnts)):
            A = cv2.contourArea(cnts[k])
            if A > min_size and A < max_size:
                com = np.mean(cv2.findNonZero(cv2.drawContours(np.zeros_like(bin).astype('uint8'), contours[0], k, 255, -1)).squeeze(),axis=0)
                if not isinstance(com,(np.ndarray,)):
                    break
                x,y,w,h= boundingBoxes[k]
                if x > 0:
                    x = x-1
                if y > 0:
                    y = y-1
                w = w+1
                h = h+1
                obj = bin[y:(y+h+1),x:(x+w+1)]

                if resize is not None:
                    obj = cv2.resize(obj,resize)

                objects.append(obj)
                info.append([i,k,A,x,y,w,h,com[0],com[1]])

    if resize is not None:
        objects = np.array(objects)[:,:,:,np.newaxis]
    info = np.array(info)

pangles = rotnet_model.predict_on_batch(objects/255).squeeze()
angles = np.argmax(angles, axis= -1)
for i in range(8):
    plt.figure()
    plt.subplot(121)
    plt.imshow(objects[i,...].squeeze())
    plt.subplot(122)
    plt.imshow(rotate(objects[i,...],-angles[i]))


# TODO: save data to excel

plt.hist(angles)
plt.hist(info[:,0],n=1600)
np.unique(info[:,0]).shape
info[:,0].shape
objects27, info27 = pipeline2.get_objects(new_input2[27:29,...], cnn, resize = (32,32), min_size = 0, max_size = 1600)

plt.imshow(new_input2[27,...].squeeze())
plt.plot(info27[info27[:,0]==1,9],info27[info27[:,0]==1,10],'r*')
plt.plot(info27[info27[:,0]==1,5],info27[info27[:,0]==1,6],'g*')

lame = [[1,2],[1],[[1,2,3],[1,23]]]
lame[0][0] = [1,2,3,4]
lame[:][-1]
len(lame)
