import tensorflow as tf
from scipy import ndimage
import numpy as np
__author__ = 'arbellea@post.bgu.ac.il'



class WeightedCELoss(object):
    def __init__(self, channel_axis, class_weights):
        self.channel_axis = channel_axis
        self.class_weights = class_weights

    def __call__(self, gt_sequence, output_sequence):
        gt_sequence = tf.squeeze(gt_sequence, self.channel_axis)
        gt_sequence = tf.cast(gt_sequence, tf.int32)
        if self.channel_axis == 2:
            output_sequence = tf.transpose(output_sequence, (0, 1, 3, 4, 2))

        onehot_gt = tf.one_hot(tf.cast(gt_sequence, tf.int32), 3)
        class_weights = tf.constant(self.class_weights)
        pixel_weights = tf.reduce_sum(onehot_gt * class_weights, axis=-1)
        gt_sequence = tf.maximum(gt_sequence, 0)
        pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_sequence, logits=output_sequence)
        weighted_pixel_loss = pixel_loss * pixel_weights
        loss = tf.reduce_mean(weighted_pixel_loss)
        return loss


class SegMeasure(object):

    def __init__(self, channel_axis, temporal=True, three_d=False, max_labels=100, foreground_class_index=1):
        self.channel_axis = channel_axis
        self.foreground_class_index = foreground_class_index
        self.max_labels = max_labels
        if temporal:
            spatial_dimensions = (2, 3) if three_d else (2, 3, 4)
        else:
            spatial_dimensions = (1, 2) if three_d else (1, 2, 3)

        self.spatial_dimensions = spatial_dimensions
        if not three_d:
            self.strel = np.zeros([3, 3, 3, 3])
            self.strel[1][1] = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:
            self.strel = np.zeros([3, 3, 3, 3, 3])
            self.strel[1][1] = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    @tf.function
    def __call__(self, gt_sequence, output_sequence):
        with tf.device('/cpu:0'):
            gt_sequence = tf.squeeze(gt_sequence, self.channel_axis)
            max_gt_label = tf.cast(tf.reduce_max(gt_sequence), tf.float32)
            output_classes = tf.argmax(output_sequence, axis=self.channel_axis)
            output_foreground = tf.equal(output_classes, self.foreground_class_index)
            output_labled, max_label = tf.py_function(self.connencted_components, inp=[output_foreground],
                                                      Tout=[tf.float32, tf.uint16])
            max_output_label = tf.reduce_max(output_labled)
            max_label = tf.maximum(max_gt_label, max_output_label)
            if tf.logical_or(tf.greater(max_label, self.max_labels), tf.less_equal(max_gt_label, 0)):
                return tf.constant(np.nan)
            else:

                gt_sequence = tf.cast(tf.expand_dims(gt_sequence, axis=-1), tf.int32)
                max_gt_label = tf.cast(max_gt_label, tf.int32)
                output_labled = tf.cast(output_labled, tf.int32)
                max_output_label = tf.cast(max_output_label, tf.int32)

                one_hot_gt = tf.cast(tf.one_hot(gt_sequence - 1, depth=max_gt_label, axis=-1), tf.bool)
                one_hot_output = tf.cast(tf.one_hot(output_labled - 1, depth=max_output_label, axis=-1), tf.bool)
                one_hot_output = tf.expand_dims(one_hot_output, axis=-1)
                intersection = tf.cast(tf.math.count_nonzero(tf.logical_and(one_hot_gt, one_hot_output),
                                                             axis=self.spatial_dimensions), tf.float32)
                area = tf.cast(tf.math.count_nonzero(one_hot_gt, axis=self.spatial_dimensions), tf.float32)
                union = tf.cast(tf.math.count_nonzero(tf.logical_or(one_hot_gt, one_hot_output),
                                                      axis=self.spatial_dimensions), tf.float32)
                overlap = tf.divide(intersection, area)
                num_objects = tf.cast(tf.math.count_nonzero(tf.greater(area, 0)), tf.float32)
                true_positive = tf.cast(tf.greater(overlap, 0.5), tf.float32)
                jaccard = tf.cast(tf.divide(intersection, union + 0.00000001), tf.float32)
                true_jaccard = tf.multiply(jaccard, true_positive)
                seg_measure = tf.reduce_sum(true_jaccard) / num_objects
                return seg_measure

    @classmethod
    def unit_test(cls):
        temporal = True
        three_d = False
        seg_measure = cls(channel_axis=4, three_d=three_d, temporal=temporal, max_labels=100, foreground_class_index=1)
        h = w = 30
        batch_size = 3
        unroll_len = 2
        gt_sequence = np.zeros((batch_size, unroll_len, h, w, 1)).astype(np.uint16)
        output_sequence = np.zeros((batch_size, unroll_len, h, w, 3)).astype(np.float32)
        output_sequence[:, :, :, :, 0] = 0.25
        objects = [(12, 20, 0, 5), (0, 9, 0, 5), (12, 20, 9, 20), (0, 9, 9, 20)]
        i = 0
        for b in range(batch_size):
            for u in range(unroll_len):
                for obj_id, (xs, xe, ys, ye) in enumerate(objects):
                    gt_sequence[b, u, ys + i:ye + i, xs + i:xe + i] = obj_id + 1
                    output_sequence[b, u, max(ys + i, 0):max(ye + i, 0), max(xs + i, 0):max(xe + i, 0), 1] = 0.5
                i += 1
        print(seg_measure(gt_sequence, output_sequence))

    def connencted_components(self, input_tensor):
        input_np = input_tensor.numpy()
        labeled, max_num = ndimage.label(input_np, structure=self.strel)
        return labeled.astype(np.uint16), np.array(max_num).astype(np.float32)


if __name__ == '__main__':
    SegMeasure.unit_test()
