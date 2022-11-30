import random
import numpy as np
import tensorflow as tf


class DataPipeline(object):
    def __init__(self, data_array, label_array, num_classes):
        self.data_array = data_array
        self.label_array = label_array
        self.num_classes = num_classes

    def augment(self, image, label):
        image = image/255.0
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, self.num_classes)
        if random.random() > 0.5:
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.5, 2.0)
            return image, label
        else:
            return image, label

    def create_pipeline(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data_array, self.label_array))
        dataset = dataset.shuffle(
            self.data_array.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.map(
            self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)
        return dataset
