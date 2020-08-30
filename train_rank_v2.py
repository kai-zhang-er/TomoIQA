import os
import tensorflow as tf
from src.datasets.load_dataset import DataGenerator
from src.net.model_v2 import VGGIQAModel
from tensorflow.keras import optimizers
from utils import ex


@ ex.capture
def scheduler(epoch):
    if epoch < rank_train_epochs* 0.4:
        return rank_learning_rate
    if epoch < rank_train_epochs * 0.8:
        return rank_learning_rate * 0.2
    return rank_learning_rate * 0.04


@ex.capture
def get_rankloss(y_true, y_pred):
    """The forward """
    num = 0
    batch = 1
    level = 6
    distortion = 3
    SepSize = batch * level
    dis = []
    margin=6
    # for the first
    for k in range(distortion):
        for i in range(SepSize * k, SepSize * (k + 1) - batch):
            for j in range(SepSize * k + int((i - SepSize * k) / batch + 1) * batch, SepSize * (k + 1)):
                dis.append(y_pred[i] - y_pred[j])
                num += 1

    diff = tf.cast(margin,tf.float32) - dis

    loss = tf.maximum(0.,diff)

    loss = tf.reduce_mean(loss)

    return loss


@ ex.main
def train():
    save_ckpt_dir = save_ckpt_dir + rank_model_name + "/"
    os.makedirs(save_ckpt_dir,exist_ok=True)

    # config the dataset loader, image size is set at 224*224
    training_generator = DataGenerator({'root_dir': root_dir, 'data_root': save_distorted_dir, 'split': 'train_info', 'im_shape': [crop_size, crop_size],'batch_size': batch_size})
    testing_generator = DataGenerator({'root_dir': root_dir, 'data_root': save_distorted_dir, 'split': 'test_info', 'im_shape': [crop_size, crop_size],'batch_size': batch_size})

    model = VGGIQAModel(is_training=True)
    model.summary()
    adam = optimizers.Adam(learning_rate=rank_learning_rate)
    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(loss=get_rankloss, optimizer=adam, metrics=[get_rankloss])
    checkpoint_path=save_ckpt_dir+'model.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1,
                                   save_weights_only=True, save_best_only=False)
    my_callbacks = [
        change_lr,
        checkpointer,
        tf.keras.callbacks.TensorBoard(log_dir=rank_logs_dir),
    ]
    model.fit_generator(generator=training_generator,epochs=rank_train_epochs,
                        validation_data=testing_generator,validation_steps=rank_test_epochs,
                        callbacks=my_callbacks)
    tf.print("finish")