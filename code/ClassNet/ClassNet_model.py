import sys
sys.path.insert(0, './code/RotNet/')
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model
from keras.optimizers import *

def fake_tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7

    return alpha*false_neg


def classnet(pretrained_weights = None,input_size = (32,32,1),classes = 1):
    # number of convolutional filters to use
    nb_filters = 128
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # model definition
    input = Input(shape=input_size)
    x = Conv2D(nb_filters, kernel_size, padding = 'same', activation='tanh')(input)
    x = Conv2D(nb_filters, kernel_size, padding = 'same', activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(nb_filters*2, kernel_size, padding = 'same', activation='tanh')(x)
    x = Conv2D(nb_filters*2, kernel_size, padding = 'same', activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(nb_filters*3, kernel_size, padding = 'same', activation='tanh')(x)
    x = Conv2D(nb_filters*3, kernel_size, padding = 'same', activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(2560, activation='tanh')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    # model compilation
    model.compile(loss = fake_tversky,
                  optimizer=Adam(lr = 1e-5),
                  metrics = ['categorical_accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
