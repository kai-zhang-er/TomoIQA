import tensorflow as tf
from tensorflow.keras import layers


class RankIQAModel(tf.keras.Model):
    def __init__(self, class_num,level_num=5):
        super(RankIQAModel, self).__init__()
        self.class_num=class_num
        self.level_num=level_num
        self.base_model = tf.keras.applications.VGG16(
            include_top=False, weights='imagenet', pooling='max', classes=self.class_num, input_shape=(224, 224, 3))
        self.flatten = layers.Flatten(name='flatten')
        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        self.rank = layers.Dense(self.class_num, name='rank')
        self.classifier = layers.Dense(self.level_num, name="classification")

    def call(self, inputs):
        x=self.base_model(inputs)
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.fc2(x)
        x_rank=self.rank(x)
        x_classification=self.classifier(x)
        return x_rank,x_classification


if __name__=="__main__":

    m=RankIQAModel(level_num=5,class_num=1)
    a = tf.random.normal((1,224, 224,3))
    y_hat,y_level=m(a)
    tf.print(tf.shape(y_hat))
    tf.print(tf.shape(y_level))