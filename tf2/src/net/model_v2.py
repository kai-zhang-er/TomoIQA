import tensorflow as tf
from tensorflow.keras import layers


def VGGIQAModel(input_shape=(224,224,3),is_training=False,class_num=1):
    inputs=layers.Input(input_shape)
    baseModel =tf.keras.applications.VGG16(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)
    x=baseModel.output
    x=layers.Dense(4096, activation='relu')(x)
    if is_training:
        x=layers.Dropout(rate=0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    if is_training:
        x = layers.Dropout(rate=0.5)(x)
    y_pred=layers.Dense(class_num)(x)
    return tf.keras.Model(inputs=inputs, outputs=y_pred)


if __name__=="__main__":

    model=VGGIQAModel(is_training=True)
    model.summary()
