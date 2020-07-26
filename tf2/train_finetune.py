import os
import tensorflow as tf
from src.datasets.load_finetune_dataset import DataGenerator
from src.net.model_v2 import VGGIQAModel
from tensorflow.keras import optimizers


experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

pretrained_model_path="experiments/TOMO/rankiqa2/model.h5"
root_dir = "/content/drive/My Drive/pyIQA/"
data_dir = "dataset/cropped/"
train_list = 'train_finetune'
test_list = 'test_finetune'
exp_name = "rankiqa2_finetune"
save_ckpt_dir = "experiments/TOMO/" + exp_name + "/"
logs_dir="experiments/finetune/"
os.makedirs(save_ckpt_dir,exist_ok=True)
train_epochs=200
test_epochs=10
learning_rate=5e-5
start_learning_rate=1e-6
batch_size=18


def scheduler(epoch):
    if epoch < train_epochs* 0.1:
        base_lr=start_learning_rate+(learning_rate-start_learning_rate)*epoch/float(train_epochs* 0.1)
        return base_lr
    if epoch < train_epochs * 0.4:
        return learning_rate * 0.5
    if epoch < train_epochs * 0.6:
        return learning_rate * 0.2
    return learning_rate * 0.04


def train():
    training_generator = DataGenerator({'root_dir': root_dir, 'data_root': data_dir, 'split': train_list, 'im_shape': [224, 224],'batch_size': batch_size})
    testing_generator = DataGenerator({'root_dir': root_dir, 'data_root': data_dir, 'split': test_list, 'im_shape': [224, 224],'batch_size': batch_size})

    model = VGGIQAModel(is_training=True)
    model.summary()
    model.load_weights(pretrained_model_path)
    adam = optimizers.Adam(learning_rate=learning_rate)

    mse = tf.keras.losses.MeanSquaredError()
    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(loss=mse, optimizer=adam, metrics=[mse])

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