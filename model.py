from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

def compute_jaccard_loss(y_true, y_pred):
    intersection = y_true * y_pred
    union = 1 - ((1 - y_true) * (1 - y_pred))
    return 1 - (K.sum(intersection) / K.sum(union))

def create_unet(pretrained_weights=None, input_shape=(128, 128, 1), lr=1e-5, decay=1e-7):
    inputs = Input(input_shape)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)
    
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    d5 = Dropout(0.5)(c5)
    
    u6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(d5))
    merge6 = concatenate([d4, u6], axis=3)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)
    
    u7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    merge7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)
    
    u8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7))
    merge8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)
    
    u9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8))
    merge9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr, decay=decay), loss='binary_crossentropy', metrics=['accuracy', compute_jaccard_loss])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model
