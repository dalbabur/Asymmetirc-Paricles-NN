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
    from scipy import signal
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

# for i in range(1):
#
    # og = np.zeros(new_input2.shape)
    # i = 0
    # load_ogs = predict_gen.flow_from_directory(
    #                 from_path,
    #                 target_size = resize,
    #                 color_mode = 'grayscale',
    #                 batch_size = 1,
    #                 class_mode = None,
    #                 shuffle = False
    # )
    #
    # while i < data_size:
    #     for d in load_ogs:
    #         og[i,...] = d.squeeze()
    #         i += 1
    #         break

    start = 100
    n_batches = 20
    for t in range(start,start+n_batches*BATCH_SIZE):
        # plt.figure(figsize=(16, 64))
        # plt.imshow(og[t,...].squeeze())
        # plt.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r*')
        # plt.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g*')
        plt.figure(figsize=(16, 64))
        plt.imshow(final_masks[t,...])
        plt.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r*')
        plt.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g*')
        plt.title(t)
#
# fig = plt.figure(figsize=(64,14))
# ims = []
# for i in range(len(d[:,0,0,0])):
#     t=i+1
#
#     fig = plt.figure(figsize=(64,14))
#
#     a1 = plt.subplot(111)
#     coms, = a1.plot(info[info[:,0]==t,9],info[info[:,0]==t,10],'r.',markersize=30)
#     boxs, = a1.plot(info[info[:,0]==t,5],info[info[:,0]==t,6],'g.',markersize=30)
#     im1 = a1.imshow(og[t,...].squeeze(),cmap='gray')
#     plt.imshow(new_input2[1,...].squeeze())
#     ims.append([im1,coms,boxs])
#
# plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
# ani = animation.ArtistAnimation(fig, ims, interval=300)
# ani.save('code/masks.avi')
# plt.show()
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
    plt.scatter(info[:,9], info[:,10], marker='o', c = info[:,11], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.hist(info[:,11])

    plt.figure(figsize = (8,6))
    [plt.hist(info[info[:,2] == d,4], alpha=0.75, label=d) for d in [0,1,2]]
    plt.legend()

# FIND PARAMETERS
distances = [50,60,70]
memories = [3,5]
total = 0
f1 = plt.figure(figsize = (32,32))

for j in range(len(memories)):
    for k in range(len(distances)):
        total = total+1
        traj, labels, dists = pipeline2.get_trajectories(info,distances[k],memories[j])
        a1 = f1.add_subplot(len(memories),len(distances),total)
        a1.title.set_text(len(traj))
        for i in range(len(traj)):
            particles = traj[i][:,2] != 2
            a1.plot(traj[i][:,9],traj[i][:,10])

traj, labels, dists = pipeline2.get_trajectories(info,200,3)
len(traj)

plt.figure(figsize = (24,12))
for i in range(len(traj)):
    plt.plot(traj[i][:,9],traj[i][:,10],'o--',label = i)
    plt.title(['COM Tracking',i])
plt.legend()

# MAKE MOVIE
ost = np.zeros(len(traj))
fig = plt.figure(figsize = (64,14))
ims = []
coms = []
rays = []
for f in range(new_input2[100:500,...].shape[0]):
f = 0
plt.plot(traj[5][:,9],traj[5][:,10],'r')
for z in range(10):
    fig = plt.figure(figsize = (64,14))
    a1 = plt.subplot(111)
    im1 = plt.imshow(new_input2[f,...].squeeze(),cmap = 'gray')
    for t in range(len(traj)):
        for p in range(len(traj[t])):
            if traj[t][p,0] == f:
                com1, = plt.plot(traj[t][p,9],traj[t][p,10],'r*',markersize=30)
                ost[t] = ost[t] + 1
                coms.append(com1)
                print(t,p)
            if ost[t] == len(traj[t]):
                ost[t] = 0
        traj1, = plt.plot(traj[t][0:ost[t].astype(int),9],traj[t][0:ost[t].astype(int),10],'r',linewidth=10)
        rays.append(traj1)
    ims.append([im1,*coms,*rays])
    coms, rays = [], []
    f += 1
plt.figure(figsize = (64,14))
ims[10]
plt.draw()
plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save('code/traj_U.avi')

plt.figure(figsize = (24,12))
signals = [3, 4,5, 9] #range(len(traj))
for i in range(len(signals)):
    plt.plot(traj[signals[i]][5:-5,9],traj[signals[i]][5:-5,10],label = signals[i])
plt.title('COM Tracking')
plt.legend()

for i in range(1):
    signals = [1,3,5]
    i = signals[np.argmax([len(traj[i][5:-5,9]) for i in signals])]
    signals.remove(i)

    fig = plt.figure(figsize = (24,12))

    x1 = traj[i][5:-5,9]
    y1 = traj[i][5:-5,10]
    a1 = traj[i][5:-5,11]
    pa1 = traj[i][5:-5,12]*600

    y_norm1 = 2*(y1-min(y1))/(max(y1)-min(y1))-1
    # plt.scatter(x1,y_norm1,c=a1, s = pa1)
    plt.plot(x1,y_norm1)

    num = 50
    h1, bins = np.histogram(x1,num)
    all_anorm = np.zeros((len(signals)+1,len(bins)+1,np.mean(h1,dtype = int)+5))
    all_anorm[:] = np.nan
    for h in range(len(x1)):
        if np.nonzero(np.isnan(all_anorm[0,np.digitize(x1,bins)[h]-1,:]).astype(int))[0].size != 0:
            all_anorm[0,np.digitize(x1,bins)[h]-1,np.min(np.nonzero(np.isnan(all_anorm[0,np.digitize(x1,bins)[h]-1,:]).astype(int)))] = a1[h]

    all_x = x1
    all_y = y_norm1
    all_a = a1
    all_pa = pa1
    for k in range(len(signals)):
        j = signals[k]
        x2 = traj[j][5:-5,9]
        y2 = traj[j][5:-5,10]
        a2 = traj[j][5:-5,11]
        pa2 = traj[i][5:-5,12]*600

        y_norm2 = 2*(y2-min(y2))/(max(y2)-min(y2))-1
        # if k == 2:
        #     y_norm2 = -y_norm2
        dx = np.mean(np.diff(x1)) # TODO: shift wrt to x, or angle???
        shift = (np.argmax(signal.correlate(y_norm1, y_norm2)) - len(y_norm2)) * dx
        # plt.scatter(x2 + shift, y_norm2,c = a2, s = pa2)
        plt.plot(x2+shift, y_norm2)

        all_x = np.append(all_x, x2+shift)
        all_y = np.append(all_y, y_norm2)
        all_a = np.append(all_a, a2)
        all_pa = np.append(all_pa, pa2)
        for h in range(len(x2)):
            if np.nonzero(np.isnan(all_anorm[k+1,np.digitize(x2,bins)[h]-1,:]).astype(int))[0].size != 0:
                all_anorm[k+1,np.digitize(x2,bins)[h]-1,np.min(np.nonzero(np.isnan(all_anorm[k+1,np.digitize(x2,bins)[h]-1,:]).astype(int)))] = a2[h]
    plt.ylim(1.25,-1.25)
    signals = [1,3,5]

    fig = plt.figure(figsize = (24,12))
    idx = np.argsort(all_x)
    wave = lambda x,a,b,c,d: a*(np.sin(b*(x +c)))+d
    SSE = lambda p: np.sum((all_y[idx]-wave(all_x[idx],*p)**2))
    opti = optimize.differential_evolution(SSE, [[-1,1],[-1,1],[-600,600],[-1,1]],seed=1)
    params, params_covariance = optimize.curve_fit(wave, all_x[idx], all_y[idx],p0 = opti.x)
    plt.scatter(all_x,all_y,c= all_a, s = all_pa)
    plt.colorbar()
    plt.ylim(1.25,-1.25)
    plt.plot(all_x[idx], wave(all_x[idx],*params))


# fit fourier
######
# from symfit import parameters, variables, sin, cos, Fit
#
# def fourier_series(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.
#
#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     a0, *a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
#     b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
#     # Construct the series
#     series = a0 + sum(ai * sin(i * f * x-bi)
#                      for i, (ai, bi) in enumerate(zip(a, b), start=1))
#     return series
#
# x, y = variables('x, y')
# w, = parameters('w')
# model_dict = {y: fourier_series(x, f=w, n=100)}
# fit = Fit(model_dict, x=all_x, y=all_y)
# fit_result = fit.execute()
#
# plt.scatter(all_x,all_y)
# plt.plot(all_x, fit.model(x=all_x, **fit_result.params).y, color='green', ls=':')



# color fitted curve by segments depending on angle
######
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
# x = np.linspace(min(all_x),max(all_x),num)
# y = wave(x,*params)
# dydx = np.nanmean(np.nanmean(all_anorm,-1),0)
#
# # Create a set of line segments so that we can color them individually
# # This creates the points as a N x 1 x 2 array so that we can stack points
# # together easily to get the segments. The segments array for line collection
# # needs to be (numlines) x (points per line) x 2 (for x and y)
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# segments.shape
# points.shape
#
# fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
#
# # Create a continuous norm to map from data points to colors
# norm = plt.Normalize(np.nanmin(dydx), np.nanmax(dydx))
# lc = LineCollection(segments, cmap='viridis', norm=norm)
# # Set the values used for colormapping
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs.add_collection(lc)
# fig.colorbar(line, ax=axs)
#
# axs.set_xlim(x.min(), x.max())
# axs.set_ylim(-1.1, 1.1)
# plt.show()



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
for i in range(100,150):
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
