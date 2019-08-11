for i in range(1):
    import sys
    sys.path.insert(0, './code/UNET/')
    from image import ImageDataGenerator, array_to_img, img_to_array, load_img,to_categorical
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

def get_objects(y_pred, class_model, rot_model, resize = None, min_size = 66,max_size=300):
    """
    takes output of UNET ( np.array of (batch_size,img_dim,img_dim,classes) ) and returns cropped particles (objects)
    and information (info) about each of them in the following order:

    info index: [0 1 2  3 4 5 6 7 8   9   10 11 12]
       meaning: [i,k,c,pc,A,x,y,w,h,com1,com2,a,pa]

       i: frame number
       k: particle number in frame
       c: class of particle
      pc: probability of class
       A: area of particle
       x: x-coordinate for minimum bounding box
       y: y-coordinate for minimum bounding box
       w: width for minimum bounding box
       h: height for minimum bounding box
    com1: x-coordinate for center of mass
    com2: y-coordinate for center of mass
       a: angle of rotation
      pa: probability of angle

    it needs a ClassNet model and a RotNet model
    it can also filter particles based on size (area)

    """
    batch_size,H,W = y_pred.shape
    objects = list()
    info = list()

    for i in range(batch_size):                                                             # loop through frames
        bin = y_pred[i,:,:][:,:,np.newaxis].astype('uint8')
        contours = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)          # find contours of particles in frame
        cnts = contours[0]
        if len(cnts)>0:                                                                     # check if there are any particles
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]                             # get the entire bounding boxes
            cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][1], reverse=False))

            for k in range(len(cnts)):                                                      # loop through found particles
                A = cv2.contourArea(cnts[k])                                                # get area of particle
                if A > min_size and A < max_size:                                           # filter by area
                    com = np.mean(cv2.findNonZero(cv2.drawContours(np.zeros_like(bin).astype('uint8'), contours[0], k, 255, -1)).squeeze(),axis=0) # find com
                    if not isinstance(com,(np.ndarray,)):
                        break
                    x,y,w,h= boundingBoxes[k]
                    if x > 0:
                        x = x-1
                    if y > 0:
                        y = y-1
                    w = w+1
                    h = h+1
                    obj = bin[y:(y+h+1),x:(x+w+1)]                                          # extract particle based on bounding box, save as obj

                    if resize is not None:
                        obj = cv2.resize(obj,resize)                                        # resize object to make sure everyhting is the same shape

                    objects.append(obj)
                    info.append([i,k,A,x,y,w,h,com[0],com[1]])

    if resize is not None:
        objects = np.array(objects)[...,np.newaxis]
    info = np.array(info)

    predictions = rot_model.predict(objects.astype('uint8'))                                 # use RotNet to calculate orientation
    info = np.append(info, np.argmax(predictions,-1)[...,np.newaxis],1)
    info = np.append(info, np.max(predictions,1)[...,np.newaxis],1)

    predictions = class_model.predict(objects.astype('uint8'))                               # use ClassNet to classify paticle
    info = np.insert(info, 2, np.argmax(predictions,1),1)
    info = np.insert(info, 3, np.max(predictions,1),1)
    return objects, info

def get_trajectories(info, distance = 65, max_memory = 3):
    """
    takes output of get_objects (info), minum "distance", and memory,  and returns a list of particles through frames (traj)
    i.e. traj[i] contains all information about the ith particle,

    "distance" is calculated like this: sqrt(x^2+(y+30)^2), so that more weight is given to the y axis
    (particles should move more on the x-axis from frame to frame, if y-axis changes too much it could be a different particle)

    memory is the number of particles from the previous frame/s that could potentially be the same as the new particle
    (mainly needed in case some frames are skipped and because there can be more than one particle per frame)

    finding the right distance and memory for each movie is very empirical as of right now

    """
    traj = []
    memory = []
    i = 0
    idx = 0
    labels = []
    all_dists = np.array([])
    while i  < max(info[:,0]).astype(int):                                                     # loop through particles (really we loop through frames, we change i inside the loop)
    # debugging print outs and plots
    # for z in range(1):

        # plt.figure(figsize=(16, 64))
        # plt.imshow(final_masks[i,...])
        # plt.plot(info[info[:,0]==i,5],info[info[:,0]==i,6],'g*')
        # plt.title(i)
        # print('-----------------------')

        dummy = list(labels)
        reset = idx
        available = np.ones(len(memory), dtype=bool)

        for f in range(sum(info[:,0] == i)):                                                   # loop through particles in the same frame
            idx = reset+f
            # print(['idx',idx,'i',i,'f',f,'total',sum(info[:,0] == i)])
            # plt.plot(info[idx,9],info[idx,10],'*')
            if memory == []:                                                                   # if memory is empty, add particle as either the first particle of the movie, or as a new particle
                traj.append(info[idx,:][np.newaxis])
                if i == 0:
                    labels.append(0)
                else:
                    labels.append(max(labels)+1)
                # print(['empty memory, new object',labels[-1]])
            else:                                                                              # if memory is not empty, find math old particle and new partilce based on distance
                dists = np.zeros(len(memory))
                for j in range(len(memory)):
                    # print([info[idx,9],info[idx,10]],[memory[j][9],memory[j][10]])
                    if available[j]:                                                           # particles can only be mathced once
                        dists[j] = ( (info[idx,9] - memory[j][9])**2 + (30*(info[idx,10] - memory[j][10]))**2)**(1/2)
                    else:
                        dists[j] = np.inf

                # print(dists)
                # print(['argmin',np.argmin(dists)])
                # print([dummy,dummy[-len(memory):]])
                label = dummy[-len(memory):][np.argmin(dists)]
                all_dists = np.append(all_dists,min(dists))

                if min(dists) < distance:                                                      # if the distance is small enough, it is the same particle
                    traj[label] = np.append(traj[label],info[idx,:][np.newaxis],axis=0)
                    # print(['append to old object',label])
                    labels.append(label)
                    available[np.argmin(dists)] = False
                else:                                                                          # if the distance is too large, they are different particles
                    traj.append(info[idx,:][np.newaxis])
                    labels.append(max(labels)+1)
                    # print(['too far, new object',labels[-1]])

        if sum(info[:,0] == i) > 0:                                                            # remove old and add new particles to the memory, based on the memory size
            if len(memory) >= max_memory:
                for f in range(sum(info[:,0] == i)):
                    if memory != []:
                        memory.pop(0)
                        # print(['memory popped'])

            for f in range(sum(info[:,0] == i)):
                memory.append(info[reset+f,:])
                # print(['added memory'])

            # print(['length memory',len(memory)])
            idx = idx+1

        i = i+1
        # print('-----------------------')
    return traj, labels, all_dists
