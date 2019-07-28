for i in range(1):
    import sys
    sys.path.insert(0, './code/UNET/')
    from image import ImageDataGenerator, array_to_img, img_to_array, load_img,to_categorical
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

def get_objects(y_pred, class_model, rot_model, resize = None, min_size = 66,max_size=300):
    """
    takes output of UNET ( np.array of (batch_size,img_dim,img_dim,classes) ) and produces
    input for RotNet (np.array of (num_objs,resize,resize,1) ), which are all bounding squares
    """
    batch_size,H,W = y_pred.shape
    objects = list()
    info = list()
    for i in range(batch_size):
        bin = y_pred[i,:,:][:,:,np.newaxis].astype('uint8')
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
        objects = np.array(objects)[...,np.newaxis]
    info = np.array(info)
    predictions = class_model.predict(objects.astype(int), 10)
    info = np.insert(info, 2, np.argmax(predictions,1),1)
    info = np.insert(info, 3, np.max(predictions,1),1)

    predictions = rot_model.predict_on_batch(objects.astype(int)) # TODO: break up into batches
    info = np.append(info, np.argmax(predictions,-1)[...,np.newaxis],1)
    info = np.append(info, np.max(predictions,1)[...,np.newaxis],1)

    return objects, info

def get_trajectories(info, distance = 25, max_memory = 10):

    traj = []
    memory = []
    i = 0
    idx = 0
    labels = []
    all_dists = np.array([])
    while i  < max(info[:,0]).astype(int):

    # debugging print outs and plots
    # for z in range(1):

        # plt.figure(figsize=(16, 64))
        # plt.imshow(final_masks[i,...])
        # plt.plot(info[info[:,0]==i,5],info[info[:,0]==i,6],'g*')
        # plt.title(i)
        # print('-----------------------')

        dummy = list(labels)
        reset = idx
        for f in range(sum(info[:,0] == i)):
            idx = idx+f
            # print(['idx',idx,'i',i,'f',f,'total',sum(info[:,0] == i)])
            # plt.plot(info[idx,9],info[idx,10],'*')
            if memory == []:
                traj.append(info[idx,:][np.newaxis])
                if i == 0:
                    labels.append(0)
                else:
                    labels.append(max(labels)+1)
                # print(['empty memory, new object',labels[-1]])
            else:
                dists = np.zeros(len(memory))
                for j in range(len(memory)):
                    # print([info[idx,9],info[idx,10]],[memory[j][9],memory[j][10]])
                    dists[j] = ( (info[idx,9] - memory[j][9])**2 + (info[idx,10] - memory[j][10])**2 + (info[idx,0] - memory[j][0])**2)**(1/2)

                # print(dists)
                # print(['argmin',np.argmin(dists)])
                # print([dummy,dummy[-len(memory):]])
                label = dummy[-len(memory):][np.argmin(dists)]
                all_dists = np.append(all_dists,min(dists))
                if min(dists) < distance:
                    traj[label] = np.append(traj[label],info[idx,:][np.newaxis],axis=0)
                    # print(['append to old object',label])
                    labels.append(label)
                else:
                    traj.append(info[idx,:][np.newaxis])
                    labels.append(max(labels)+1)
                    # print(['too far, new object',labels[-1]])

        if sum(info[:,0] == i) > 0:
            if len(memory) >= max_memory:
                for f in range(sum(info[:,0] == i)):
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
