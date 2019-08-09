# TODO: here goes movie making, plotting cos v y/H, movie with "fixed" X, random sample to check, object properties
# will clean up predict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix


def display_segmentation(imgs,random = True, frames = None):
    if random:
        plt.figure()
        plt.imshow(imgs[np.random.randint(imgs.shape[0]-1),...].squeeze())
    else:
        for i in range(len(frames)):
            plt.figure()
            plt.imshow(imgs[frames[i],...].squeeze())
    return

def display_rotation(objects,random = True, frames = None):
    if random:
        plt.figure()
        plt.imshow(objects[np.random.randint(objects.shape[0]-1),...].squeeze())
    else:
        for i in range(len(frames)):
            plt.figure()
            plt.imshow(objects[frames[i],...].squeeze())
    return

def traj_movie(imgs = None, traj = None, frames = []):
    ost = np.zeros(len(traj))
    fig = plt.figure(figsize = (64,14))
    ims = []
    coms = []
    rays = []
    for f in range(frames[0],frames[1]):
        a1 = plt.subplot(111)
        im1 = plt.imshow(imgs[f,...].squeeze(),cmap = 'gray')
        for t in range(len(traj)):
            for p in range(len(traj[t])):
                if traj[t][p,0] == f:
                    com1, = plt.plot(traj[t][p,9],traj[t][p,10],'r*',markersize=30)
                    ost[t] = ost[t] + 1
                    coms.append(com1)
                if ost[t] == len(traj[t]):
                    ost[t] = 0
            traj1, = plt.plot(traj[t][0:ost[t].astype(int),9],traj[t][0:ost[t].astype(int),10],'r',linewidth=10)
            rays.append(traj1)
        ims.append([im1,*coms,*rays])
        coms, rays = [], []

    plt.draw()
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    # ani.save('code/traj_U.avi')
    return ani

def object_summary(info):
    # [0 1 2  3 4 5 6 7 8   9   10 11 12]
    # [i,k,c,pc,A,x,y,w,h,com1,com2,a,pa]

    plt.figure(figsize = (8,6))
    plt.scatter(info[:,2], info[:,3], marker='o', c = info[:,2], s = info[:,4])
    plt.colorbar()

    plt.figure(figsize = (16,8))
    plt.scatter(np.arange(len(info)), info[:,9], marker='o', c = info[:,2], s = info[:,4])
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
    return

def fixed_comx(imgs,mask,traj, figsize = (7,14), width = 50, markersize  = 20):
    # imgs = new_input2
    # imgs.shape
    # traj = traj[4]
    fig = plt.figure(figsize=figsize)
    ims = []
    for i in range(len(traj)):

        f = traj[i,0].astype(int)
        ax1 = plt.subplot(111)

        com1, = ax1.plot(width/2,traj[i,10],'r.',markersize=markersize)

        minx = np.round(traj[i,9]-width/2).astype(int)
        maxx = np.round(traj[i,9]+width/2).astype(int)

        if minx < 0:
            minx = 0
        if maxx > imgs.shape[2]:
            maxx = imgs.shape[2]

        im1 = ax1.imshow(imgs[f,:,minx:maxx].squeeze(),cmap='gray')
        im2 = ax1.imshow(mask[f,:,minx:maxx].squeeze(),'jet', alpha = 0.35)
        ims.append([im1,im2,com1])
        plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'
        ani = animation.ArtistAnimation(fig, ims, interval=100);
        # ani.save('code/masks.avi')
    return ani

def plot_angle():
    return

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
