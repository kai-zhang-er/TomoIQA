import os

import tensorflow as tf

from src.datasets.load_dataset import DataGenerator
from src.net.model_v2 import VGGIQAModel
from tensorflow.keras import optimizers


experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

pretrained_model_path="experiments/vgg_models/"+'vgg16_weights.npz'
root_dir = "/content/drive/My Drive/pyIQA/"
data_dir = "TOMO/"
train_list = 'train_info'
test_list = 'test_info'
exp_name = "rankiqa2"
save_ckpt_dir = "experiments/TOMO/" + exp_name + "/"
logs_dir="experiments/train/"
os.makedirs(save_ckpt_dir,exist_ok=True)
train_epochs=200
test_epochs=10
learning_rate=5e-5
batch_size=18


def scheduler(epoch):
    if epoch < train_epochs* 0.4:
        return learning_rate
    if epoch < train_epochs * 0.8:
        return learning_rate * 0.2
    return learning_rate * 0.04


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


def train():
    training_generator = DataGenerator({'root_dir': root_dir, 'data_root': data_dir, 'split': train_list, 'im_shape': [224, 224],'batch_size': batch_size})
    testing_generator = DataGenerator({'root_dir': root_dir, 'data_root': data_dir, 'split': test_list, 'im_shape': [224, 224],'batch_size': batch_size})

    model = VGGIQAModel(is_training=True)
    model.summary()
    adam = optimizers.Adam(learning_rate=learning_rate)
    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(loss=get_rankloss, optimizer=adam, metrics=[get_rankloss])
    checkpoint_path=save_ckpt_dir+'model.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1,
                                   save_weights_only=True, save_best_only=False)
    my_callbacks = [
        change_lr,
        checkpointer,
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
    ]
    model.fit_generator(generator=training_generator,epochs=train_epochs,
                        validation_data=testing_generator,validation_steps=test_epochs,
                        callbacks=my_callbacks)
    tf.print("finish")

if __name__=="__main__":
    train()