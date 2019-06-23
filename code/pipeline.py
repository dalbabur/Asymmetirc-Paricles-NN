for i in range(1):
    import sys
    sys.path.insert(0, './code/UNET/')
    from image import ImageDataGenerator, array_to_img, img_to_array, load_img,to_categorical
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

def get_objects(y_pred, resize = None):
    """
    takes output of UNET ( np.array of (batch_size,img_dim,img_dim,classes) ) and produces
    input for RotNet (np.array of (num_objs,resize,resize,1) ), which are all bounding squares
    """
    batch_size,H,W,classes = y_pred.shape
    objects = list()
    for i in range(batch_size):
        for j in range(classes-2): # don't care about background, don't care about UFOs
            bin = y_pred[i,:,:,j+1][:,:,np.newaxis].astype('uint8')
            contours = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for k in range(len(contours[0])):
                print(cv2.contourArea(contours[0][k]))
                if cv2.contourArea(contours[0][k]) > 66:
                    x,y,w,h = cv2.boundingRect((contours[0][k]))
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
    if resize is not None:
        objects = np.array(objects)[:,:,:,np.newaxis]
    return objects
