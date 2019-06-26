for i in range(1):
    import sys
    sys.path.insert(0, './code/UNET/')
    from image import array_to_img, img_to_array, load_img
    from keract import *
    import matplotlib.pyplot as plt
    from UNET_model import *
    import numpy as np

img = load_img('code/UNET/data/test/img/frames/14.tif', color_mode = 'grayscale',target_size=(64,640))
img = img_to_array(img)/255
img = img[np.newaxis]
mask = load_img('code/UNET/data/test/mask/frames/14.tif', color_mode = 'grayscale',target_size=(64,640))
mask = img_to_array(mask,dtype='uint8').squeeze()

for i in range(1):
    plt.figure(figsize=(16,64))
    plt.imshow(img.squeeze())
    plt.figure(figsize=(16,64))
    plt.imshow(mask)


model = unet(pretrained_weights = 'code/UNET/UNET.h5',classes=4)
act = get_activations(model,img,'conv2d_23')
display_activations(act)

final = list(act.values())[0]
final = np.argmax(final,axis=-1).squeeze()
for i in range(1):
    plt.subplot(1,3,1)
    plt.imshow(img.squeeze())
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.imshow(final)

all_act = get_activations(model,img)
display_activations(all_act, save = True)
