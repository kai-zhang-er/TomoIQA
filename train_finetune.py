import os
import tensorflow as tf
from src.datasets.load_finetune_dataset import DataGenerator
from src.net.model_v2 import VGGIQAModel
from tensorflow.keras import optimizers
from utils import ex


@ ex.capture
def scheduler(epoch):
    if epoch < finetunne_train_epochs* 0.1:
        base_lr=finetune_start_learning_rate+(finetune_learning_rate-finetune_start_learning_rate)*epoch/float(finetune_train_epochs* 0.1)
        return base_lr
    if epoch < finetune_train_epochs * 0.4:
        return finetune_learning_rate * 0.5
    if epoch < finetune_train_epochs * 0.6:
        return finetune_learning_rate * 0.2
    return finetune_learning_rate * 0.04


@ex.main
def train():
    training_generator = DataGenerator({'root_dir': root_dir, 'data_root': save_distorted_dir, 'split': 'train_finetune', 'im_shape': [224, 224],'batch_size': batch_size})
    testing_generator = DataGenerator({'root_dir': root_dir, 'data_root': save_distorted_dir, 'split': 'test_finetune', 'im_shape': [224, 224],'batch_size': batch_size})

    # load rank model
    model = VGGIQAModel(is_training=True)
    model.summary()
    model.load_weights(save_ckpt_dir + rank_model_name + "/")

    adam = optimizers.Adam(learning_rate=finetune_learning_rate)

    mse = tf.keras.losses.MeanSquaredError()
    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(loss=mse, optimizer=adam, metrics=[mse])

    checkpoint_dir=save_ckpt_dir+finetune_model_name+"/"
    os.makedirs(checkpoint_dir,exist_ok=True)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir+"model.h5", verbose=1,
                                   save_weights_only=True, save_best_only=False)
    my_callbacks = [
        change_lr,
        checkpointer,
        tf.keras.callbacks.TensorBoard(log_dir=finetune_logs_dir),
    ]
    model.fit_generator(generator=training_generator,epochs=finetune_train_epochs,
                        validation_data=testing_generator,validation_steps=finetune_test_epochs,
                        callbacks=my_callbacks)
    tf.print("finish")
