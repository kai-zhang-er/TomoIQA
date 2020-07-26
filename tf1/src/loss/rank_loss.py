#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created rank_loss.py by rjw at 19-1-15 in WHU.
"""

import tensorflow as tf

class Rank_loss():
    """Layer of Efficient Siamese loss function."""

    def __init__(self):
        self.margin = 6
        print('*********************** SETTING UP****************')
        pass

    @tf.function
    def get_rankloss(self, p_hat):
        """The forward """
        Num = 0
        batch = 1
        level = 6
        distortion = 3
        SepSize = batch * level
        dis = []
        # for the first
        for k in range(distortion):
            for i in range(SepSize * k, SepSize * (k + 1) - batch):
                for j in range(SepSize * k + int((i - SepSize * k) / batch + 1) * batch, SepSize * (k + 1)):
                    dis.append(p_hat[i] - p_hat[j])
                    Num += 1

        diff = tf.cast(self.margin,tf.float32) - dis

        loss = tf.maximum(0.,diff)

        loss = tf.reduce_mean(loss)

        return loss

if __name__=="__main__":
    import numpy as np

    y_hat = np.random.random([24])
    rank_loss = Rank_loss()
    loss= rank_loss.get_rankloss(y_hat)
    tf.print(loss)

