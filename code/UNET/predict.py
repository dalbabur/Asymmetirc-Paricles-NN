for i in range(1):
    from image import ImageDataGenerator, save_img
    import matplotlib.pyplot as plt
    import cv2
    from model import *
    from keras.callbacks import *
    import numpy as np
    from keract import *

data_size = 4540
BATCH_SIZE = 1
classes = 4
resize = (64,64)
n_batches = data_size/BATCH_SIZE
to_folder = '/data/predicted/mask/'
path = 'C:/Users/Diego/Documents/MATLAB/JHU/HUR/asymmetricParticles/code/UNET'

predict_gen = ImageDataGenerator(rescale = 1./255)
predict_img_generator = predict_gen.flow_from_directory(
                'data/predicted/img',
                target_size = resize,
                color_mode = 'grayscale',
                batch_size = BATCH_SIZE,
                class_mode = None,
                shuffle = False
)

model = unet(pretrained_weights = 'model2.h5', classes =classes)
predictions = model.predict_generator(predict_img_generator, steps = n_batches)


# INPUT IS NOW 64x640, not 64x64, no need to put images together again.



# input = np.zeros((data_size,)+resize+(1,))
# for j in range(1):
#     i = 0
#     for p in predict_img_generator:
#         input[i*BATCH_SIZE:((BATCH_SIZE)+i*BATCH_SIZE),...] = p
#         i += 1
#         if i == n_batches:
#             break

final_masks = np.argmax(predictions,axis=-1)
full_masks = np.zeros((int(data_size/10),resize[0],resize[1]*10))
# full_input = np.zeros(full_masks.shape)

for i in range(int(data_size/10)):
    for j in range(10):
        full_masks[i,:,resize[1]*j:(resize[1]*(j+1))] = final_masks[i*10+j]
        # full_input[i,:,resize[1]*j:(resize[1]*(j+1))] = input[i*10+j].squeeze()

for i in range(int(data_size/10)):
    save_img(path+to_folder+str(i)+'.tif',full_masks[i][:,:,np.newaxis],scale=False)
    save_img(path+to_folder+str(i)+'_scaled.tif',full_masks[i][:,:,np.newaxis],scale=True)
    # plt.figure(figsize=(16, 32))
    # plt.imshow(full_input[i])
    # plt.figure(figsize=(16, 32))
    # plt.imshow(full_masks[i])
