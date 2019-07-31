import sys
sys.path.insert(0, './code/RotNet/')
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise
from keras.models import Model
from utils import angle_error_regression, RotNetDataGenerator, binarize_images, rotate, angle_error

def rotnet(pretrained_weights = None,input_size = (32,32,1),classes = 1):
    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # model definition
    input = Input(shape=input_size)
    x = Conv2D(nb_filters, kernel_size, activation='relu')(input)
    x = Conv2D(nb_filters, kernel_size, activation='relu')(x)
    # x = GaussianNoise(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)

    # model compilation
    model.compile(loss = 'categorical_crossentropy',
                  optimizer='adam',
                  metrics = [angle_error,'categorical_accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
