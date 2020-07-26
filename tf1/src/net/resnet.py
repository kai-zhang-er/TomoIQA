import tensorflow as tf
import numpy as np

def _resnet_block_v1(inputs, filters, stride, projection, stage, blockname, TRAINING):
    # defining name basis
    conv_name_base = 'res' + str(stage) + blockname + '_branch'
    bn_name_base = 'bn' + str(stage) + blockname + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):
        if projection:
            shortcut = tf.layers.conv2d(inputs, filters, (1, 1),
                                        strides=(stride, stride),
                                        name=conv_name_base + '1',
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        reuse=tf.AUTO_REUSE, padding='same',
                                        data_format='channels_first')
            shortcut = tf.layers.batch_normalization(shortcut, axis=1, name=bn_name_base + '1',
                                                     training=TRAINING, reuse=tf.AUTO_REUSE)
        else:
            shortcut = inputs

        outputs = tf.layers.conv2d(inputs, filters,
                                   kernel_size=(3, 3),
                                   strides=(stride, stride),
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   name=conv_name_base + '2a', reuse=tf.AUTO_REUSE, padding='same',
                                   data_format='channels_first')
        outputs = tf.layers.batch_normalization(outputs, axis=1, name=bn_name_base + '2a',
                                                training=TRAINING, reuse=tf.AUTO_REUSE)
        outputs = tf.nn.relu(outputs)

        outputs = tf.layers.conv2d(outputs, filters,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   name=conv_name_base + '2b', reuse=tf.AUTO_REUSE, padding='same',
                                   data_format='channels_first')
        outputs = tf.layers.batch_normalization(outputs, axis=1, name=bn_name_base + '2b',
                                                training=TRAINING, reuse=tf.AUTO_REUSE)
        outputs = tf.add(shortcut, outputs)
        outputs = tf.nn.relu(outputs)
    return outputs


def _resnet_block_v2(inputs, filters, stride, projection, stage, blockname, TRAINING):
    # defining name basis
    conv_name_base = 'res' + str(stage) + blockname + '_branch'
    bn_name_base = 'bn' + str(stage) + blockname + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):
        shortcut = inputs
        outputs = tf.layers.batch_normalization(inputs, axis=1, name=bn_name_base + '2a',
                                                training=TRAINING, reuse=tf.AUTO_REUSE)
        outputs = tf.nn.relu(outputs)
        if projection:
            shortcut = tf.layers.conv2d(outputs, filters, (1, 1),
                                        strides=(stride, stride),
                                        name=conv_name_base + '1',
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        reuse=tf.AUTO_REUSE, padding='same',
                                        data_format='channels_first')
            shortcut = tf.layers.batch_normalization(shortcut, axis=1, name=bn_name_base + '1',
                                                     training=TRAINING, reuse=tf.AUTO_REUSE)

        outputs = tf.layers.conv2d(outputs, filters,
                                   kernel_size=(3, 3),
                                   strides=(stride, stride),
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   name=conv_name_base + '2a', reuse=tf.AUTO_REUSE, padding='same',
                                   data_format='channels_first')

        outputs = tf.layers.batch_normalization(outputs, axis=1, name=bn_name_base + '2b',
                                                training=TRAINING, reuse=tf.AUTO_REUSE)
        outputs = tf.nn.relu(outputs)
        outputs = tf.layers.conv2d(outputs, filters,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   name=conv_name_base + '2b', reuse=tf.AUTO_REUSE, padding='same',
                                   data_format='channels_first')

        outputs = tf.add(shortcut, outputs)
    return outputs


def inference(images, training, filters, n, ver):
    """Construct the resnet model
    Args:
      images: [batch*channel*height*width]
	  training: boolean
	  filters: integer, the filters of the first resnet stage, the next stage will have filters*2
	  n: integer, how many resnet blocks in each stage, the total layers number is 6n+2
	  ver: integer, can be 1 or 2, for resnet v1 or v2
    Returns:
      Tensor, model inference output
    """
    # Layer1 is a 3*3 conv layer, input channels are 3, output channels are 16
    inputs = tf.layers.conv2d(images, filters=16, kernel_size=(3, 3), strides=(1, 1),
                              name='conv1', reuse=tf.AUTO_REUSE, padding='same', data_format='channels_first')

    # no need to batch normal and activate for version 2 resnet.
    if ver == 1:
        inputs = tf.layers.batch_normalization(inputs, axis=1, name='bn_conv1',
                                               training=training, reuse=tf.AUTO_REUSE)
        inputs = tf.nn.relu(inputs)

    for stage in range(3):
        stage_filter = filters * (2 ** stage)
        for i in range(n):
            stride = 1
            projection = False
            if i == 0 and stage > 0:
                stride = 2
                projection = True
            if ver == 1:
                inputs = _resnet_block_v1(inputs, stage_filter, stride, projection,
                                          stage, blockname=str(i), TRAINING=training)
            else:
                inputs = _resnet_block_v2(inputs, stage_filter, stride, projection,
                                          stage, blockname=str(i), TRAINING=training)

    # only need for version 2 resnet.
    if ver == 2:
        inputs = tf.layers.batch_normalization(inputs, axis=1, name='pre_activation_final_norm',
                                               training=training, reuse=tf.AUTO_REUSE)
        inputs = tf.nn.relu(inputs)

    axes = [2, 3]
    inputs = tf.reduce_mean(inputs, axes, keep_dims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')

    inputs = tf.reshape(inputs, [-1, filters * (2 ** 2)])
    inputs = tf.layers.dense(inputs=inputs, units=1, name='dense1', reuse=tf.AUTO_REUSE)
    return inputs

class ResNet_v2:
    def __init__(self,images):
        self.images=images
        self.filters = 16  # the first resnet block filter number
        self.n = 5  # the basic resnet block number, total network layers are 6n+2
        self.ver = 2  # the resnet block version

    def inference(self,training):
        return inference(self.images,training,self.filters,self.n,self.ver)

if __name__=="__main__":
    with tf.Graph().as_default(), tf.Session() as sess:
        images = np.random.random([4, 224, 224, 3])
        img = tf.cast(images, tf.float32)
        filters = 16  # the first resnet block filter number
        n = 5  # the basic resnet block number, total network layers are 6n+2
        ver = 2  # the resnet block version
        # Get the inference logits by the model
        result = inference(img, True, filters, n, ver)
        print(result.shape)
