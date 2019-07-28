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
    from scipy import optimize
    import matplotlib.animation as animation
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

unet_model = unet(pretrained_weights = 'code/UNET/weights/UNET_bin.h5', classes = classes)
predictions = unet_model.predict_generator(predict_img_generator, steps = n_batches)
final_masks = np.argmax(predictions,axis=-1)
new_input = to_categorical(final_masks,num_classes = classes)
new_input2 = (final_masks>0).astype(int);

plt.imshow(new_input2[np.random.randint(data_size-1),...].squeeze())
plt.imshow(new_input2[2,...].squeeze())
a = new_input2[1,...][np.newaxis]

# for i in range(1):
#
    og = np.zeros(new_input2.shape)
    i = 0
    load_ogs = predict_gen.flow_from_directory(
                    from_path,
                    target_size = resize,
                    color_mode = 'grayscale',
                    batch_size = 1,
                    class_mode = None,
                    shuffle = False
    )

    while i < data_size:
        for d in load_ogs:
            og[i,...] = d.squeeze()
            i += 1
            break

#     start = 0
#     n_batches = 20
#     for t in range(start,start+n_batches*BATCH_SIZE):
#         # plt.figure(figsize=(16, 64))
#         # plt.imshow(og[t,...].squeeze())
#         # plt.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r*')
#         # plt.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g*')
#         plt.figure(figsize=(16, 64))
#         plt.imshow(final_masks[t,...])
#         plt.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r*')
#         plt.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g*')
#         plt.title(t)
#
fig = plt.figure(figsize=(64,14))
ims = []
for i in range(len(d[:,0,0,0])):
    t=i+1

    fig = plt.figure(figsize=(64,14))

    a1 = plt.subplot(111)
    coms, = a1.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r.',markersize=30)
    boxs, = a1.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g.',markersize=30)
    im1 = a1.imshow(og[t,...].squeeze(),cmap='gray')
    plt.imshow(new_input2[1,...].squeeze())
    ims.append([im1,coms,boxs])

plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
ani = animation.ArtistAnimation(fig, ims, interval=300)
ani.save('code/masks2.avi')
plt.show()
#
# if save_masks:
#  file_names = listdir(from_path+'/frames/')
#  if not os.path.exists(path+to_folder):
#     os.makedirs(path+to_folder)
#  for i in range(int(data_size)):
#     save_img(path+to_folder+file_names[i],final_masks[i][:,:,np.newaxis])

cnn = classnet(classes = 3 ,pretrained_weights = 'code/ClassNet/weights/ClassNet2.h5',input_size = (32,32,1))
rnn = rotnet(pretrained_weights = 'code/RotNet/weights/RotNet_wAugmentation.h5', classes = 360)
objects, info = pipeline2.get_objects(new_input2, cnn, rnn, resize = (32,32), min_size = 0, max_size = 1600)

plt.imshow(objects[np.random.randint(len(objects)-1),...].squeeze())
info.shape

# [0 1 2  3 4 5 6 7 8   9   10 11 12]
# [i,k,c,pc,A,x,y,w,h,com1,com2,a,pa]

for ii in range(1):
    plt.figure(figsize = (8,6))
    plt.scatter(info[:,2], info[:,3], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(np.arange(len(objects)), info[:,9], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(info[:,5], info[:,6], marker='o', c = info[:,0], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(info[:,9], info[:,10], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.hist(info[:,11])

    plt.figure(figsize = (8,6))
    [plt.hist(info[info[:,2] == d,4], alpha=0.75, label=d) for d in [0,1,2]]
    plt.legend()

# distances = [54]
# memories = [3]
# total = 0
# f1 = plt.figure(figsize = (32,32))
#
# for j in range(len(memories)):
#     for k in range(len(distances)):
#         total = total+1
#         traj, labels, dists = pipeline2.get_trajectories(info,distances[k],memories[j])
#         a1 = f1.add_subplot(6,6,total)
#         a1.title.set_text(len(traj))
#         for i in range(len(traj)):
#             particles = traj[i][:,2] != 2
#             a1.plot(traj[i][:,9],traj[i][:,10])

traj, labels, dists = pipeline2.get_trajectories(info,54,3)

plt.figure(figsize = (24,12))
for i in range(len(traj)):
    plt.plot(traj[i][5:-5,9],traj[i][5:-5,10],label = i)
plt.legend()


ost = np.zeros(len(traj))
fig = plt.figure(figsize = (64,14))
ims = []
coms = []
rays = []
for f in range(og.shape[0]):
    a1 = plt.subplot(111)
    im1 = a1.imshow(og[f,:,:].squeeze(),cmap = 'gray')
    for t in range(len(traj)):
        for p in range(len(traj[t])):
            if traj[t][p,0] == f:
                com1, = a1.plot(traj[t][p,9],traj[t][p,10],'r*',markersize=30)
                ost[t] = ost[t] + 1
                coms.append(com1)
            if ost[t] == len(traj[t]):
                ost[t] = 0
        traj1, = a1.plot(traj[t][0:ost[t].astype(int),9],traj[t][0:ost[t].astype(int),10],'r',linewidth=10)
        rays.append(traj1)
    ims.append([im1,*coms,*rays])
    coms, rays = [], []

plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
ani = animation.ArtistAnimation(fig, ims, interval=200)
ani.save('code/traj.avi')

for i in range(len(traj)):
    if len(traj[i][5:-5,9]) >= 4:

        fig = plt.figure(figsize = (24,12))
        from scipy.interpolate import splev, splrep
        x = traj[i][5:-5,9]
        y = traj[i][5:-5,10]
        spl = splrep(x, y)
        x2 = np.linspace(min(traj[i][5:-5,9]), max(traj[i][5:-5,9]), 200)
        y2 = splev(x2, spl)

        wave = lambda x,a,b,c,d: a*(np.sin(b*x +c)) + d
        SSE = lambda p: np.sum((y-wave(x,*p)**2))
        opti = optimize.differential_evolution(SSE, [[0,100],[-100,100],[-100,100],[0, 100]],seed=1)
        params, params_covariance = optimize.curve_fit(wave, x, y, p0 = opti.x)

        SSE = lambda p: np.sum((y2-wave(x2,*p)**2))
        opti1 = optimize.differential_evolution(SSE, [[0,100],[-100,100],[-100,100],[0, 100]],seed=1)
        params1, params_covariance = optimize.curve_fit(wave, x2, y2, p0 = opti1.x)

        plt.plot(x, y, 'o--', x2, y2,'--',x2,wave(x2,*params),x2,wave(x2,*params1))

from scipy import signal
signals = [1,3,5]
i = signals[np.argmax([len(traj[i][5:-5,9]) for i in signals])]
signals.remove(i)

fig = plt.figure(figsize = (24,12))

x1 = traj[i][5:-5,9]
y1 = traj[i][5:-5,10]
y_norm1 = 2*(y1-min(y1))/(max(y1)-min(y1))-1
plt.plot(x1,y_norm1,'o--')

all_x = x1
all_y = y_norm1

for j in signals:
    x2 = traj[j][5:-5,9]
    y2 = traj[j][5:-5,10]
    y_norm2 = 2*(y2-min(y2))/(max(y2)-min(y2))-1

    dx = np.mean(np.diff(x1))
    shift = (np.argmax(signal.correlate(y_norm1, y_norm2)) - len(y_norm2)) * dx
    plt.plot(x2 + shift, y_norm2,'o--')

    all_x = np.append(all_x, x2+shift)
    all_y = np.append(all_y, y_norm2)
signals = [1,3,5]

fig = plt.figure(figsize = (24,12))
idx = np.argsort(all_x)
wave = lambda x,a,b,c,d: a*(np.sin(b*x +c))+d
SSE = lambda p: np.sum((all_y[idx]-wave(all_x[idx],*p)**2))
opti = optimize.differential_evolution(SSE, [[-1,1],[-1,1],[-1,1],[-1,1]],seed=1)
params, params_covariance = optimize.curve_fit(wave, all_x[idx], all_y[idx],p0 = opti.x)
plt.scatter(all_x,all_y)
plt.plot(all_x[idx], wave(all_x[idx],*params))

fig = plt.figure(figsize = (24,12))
idx = np.argsort(x1)
wave = lambda x,a,b,c,d: a*(np.sin(b*x +c))+d
SSE = lambda p: np.sum((y_norm1[idx]-wave(x1[idx],*p)**2))
opti = optimize.differential_evolution(SSE, [[-1,1],[-1,1],[-1,1],[-1,1]],seed=1)
params, params_covariance = optimize.curve_fit(wave, x1[idx], y_norm1[idx],p0 = opti.x)
plt.scatter(x1,y_norm1)
plt.plot(x1[idx], wave(x1[idx],*params))


rotnet_model = rotnet(pretrained_weights = 'code/RotNet/weights/RotNet_wAugmentation.h5', classes = 360)

# might have to break this into smaller batches
angles = rotnet_model.predict_on_batch(objects).squeeze()
angles = np.argmax(angles,axis=-1)
objects[1,...].shape
np.sum([len(traj[i]) for i in range(len(traj))])

for i in range(50):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(objects[i,:,:].squeeze())
    plt.title([info[i,2],info[i,3]])
    plt.subplot(1,2,2)
    plt.imshow(rotate(objects[i,...].squeeze(),-angles[i]))
    plt.title([-angles[i],info[i,9],info[i,10]])



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
