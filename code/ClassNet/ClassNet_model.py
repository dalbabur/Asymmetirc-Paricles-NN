import sys
sys.path.insert(0, './code/RotNet/')
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model
from keras.optimizers import *

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    gamma = 1/3

    return alpha*false_neg

def tversky_loss(y_true, y_pred):
    alpha = 4
    beta  = 0.3
    gamma = 1/1.5
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, (0,1)) + K.epsilon()
    den = num + alpha*K.sum(p0*g1,(0,1)) + beta*K.sum(p1*g0,(0,1))+K.epsilon()

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return K.pow(1-T,gamma)

def dice_coef_multilabel(numLabels=4):
    def lossFunc(y_true,y_pred):
        dice=0
        for index in range(1,numLabels-1):
            dice -= dice_coef(y_true[:,index], y_pred[:,index]) # could multiply here by loss
        return dice
    return lossFunc

def dice_coef(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true[...,1:])
    y_pred_f = K.batch_flatten(y_pred[...,1:])
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + K.epsilon()
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + K.epsilon()
    return K.mean(intersection / union)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true,y_pred)

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
    dice_dice = dice_coef_multilabel(classes)
    # model compilation
    model.compile(loss = tversky,
                  optimizer=Adam(lr = 1e-5),
                  metrics = ['categorical_accuracy',tversky_loss,dice_coef_loss,dice_dice])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
