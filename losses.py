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
        gt_sequence_valid = tf.cast(tf.greater(gt_sequence, -1), tf.float32)
        gt_sequence = tf.cast(gt_sequence, tf.int32)
        if self.channel_axis == 2:
            output_sequence = tf.transpose(output_sequence, (0, 1, 3, 4, 2))

        onehot_gt = tf.one_hot(tf.cast(gt_sequence, tf.int32), 3)
        class_weights = tf.constant(self.class_weights)
        pixel_weights = tf.reduce_sum(onehot_gt * class_weights, axis=-1)
        gt_sequence = tf.maximum(gt_sequence, 0)
        pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_sequence, logits=output_sequence)
        weighted_pixel_loss = pixel_loss * pixel_weights * gt_sequence_valid
        loss = tf.reduce_sum(weighted_pixel_loss) / (tf.reduce_sum(gt_sequence_valid) + 0.00001)
        return loss

def seg_measure(channel_axis, three_d=False, foreground_class_index=1):
    if not three_d:
        strel = np.zeros([3, 3])
        strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        strel = np.zeros([3, 3, 3, 3, 3])
        strel[1][1] = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    def connencted_components(input_np):
        labeled = np.zeros_like(input_np, dtype=np.uint16)
        max_num = 0
        for d1, images in enumerate(input_np):
            for d2, image in enumerate(images):
                labeled_image, max_num_temp = ndimage.label(image, structure=strel)
                labeled[d1, d2] = labeled_image
                max_num = np.maximum(max_num, max_num_temp)

        return labeled, np.array(max_num).astype(np.float32)

    def seg_numpy(gt, seg):
        gt_labled, _ = connencted_components(gt.numpy())
        seg_labled, _ = connencted_components(seg.numpy())

        all_iou = []
        for gt1, seg1 in zip(gt_labled, seg_labled):
            for gt, seg in zip(gt1, seg1):
                for this_label in np.unique(gt):
                    if this_label == 0:
                        continue
                    all_iou.append(0.)
                    bw = gt == this_label
                    l_area = np.sum(bw).astype(np.float32)
                    overlaping_inds = seg[bw]
                    for s in np.unique(overlaping_inds):
                        if s == 0:
                            continue
                        intersection = np.sum(overlaping_inds == s).astype(np.float32)
                        overlap = intersection / l_area
                        if overlap > 0.5:
                            s_area = np.sum(seg == s).astype(np.float32)
                            iou = intersection / (l_area + s_area - intersection)
                            all_iou[-1] = iou
        if not len(all_iou):
            return np.nan
        return np.mean(all_iou)

    # @tf.function
    def calc_seg(gt_sequence, output_sequence):
        with tf.device('/cpu:0'):
            gt_sequence = tf.squeeze(gt_sequence, channel_axis)
            gt_valid = tf.cast(tf.greater(gt_sequence, -1), tf.float32)
            gt_sequence = gt_sequence * gt_valid
            gt_fg = tf.equal(tf.cast(gt_sequence, tf.float32), foreground_class_index)
            output_classes = tf.argmax(output_sequence, axis=channel_axis)
            output_foreground = tf.equal(output_classes, foreground_class_index)
            seg_measure_value = tf.py_function(seg_numpy, inp=[gt_fg, output_foreground], Tout=[tf.float32])
            return seg_measure_value

    return calc_seg


def seg_measure_unit_test():
    three_d = False
    channel_axis = 4 if not three_d else 5
    calc_seg_meas = seg_measure(channel_axis=channel_axis, three_d=three_d,
                                foreground_class_index=1)
    h = w = d = 30
    batch_size = 3
    unroll_len = 2
    if three_d:
        gt_sequence = np.zeros((batch_size, unroll_len, d, h, w, 1)).astype(np.float32)
        output_sequence = np.zeros((batch_size, unroll_len, d, h, w, 3)).astype(np.float32)
        output_sequence[:, :, :, :, 0] = 0.25
    else:
        gt_sequence = np.zeros((batch_size, unroll_len, h, w, 1)).astype(np.float32)
        output_sequence = np.zeros((batch_size, unroll_len, h, w, 3)).astype(np.float32)
        output_sequence[:, :, :, :, 0] = 0.25
    objects = [(12, 20, 0, 5), (0, 9, 0, 5), (12, 20, 9, 20), (0, 9, 9, 20)]
    i = 0
    for b in range(batch_size):
        for u in range(unroll_len):
            for obj_id, (xs, xe, ys, ye) in enumerate(objects):
                gt_sequence[b, u, ys + i:ye + i, xs + i:xe + i] = obj_id + 1
                output_sequence[b, u, max(ys + i + 2, 0):max(ye + i, 0), max(xs + i, 0):max(xe + i, 0), 1] = 0.5
            i += 1
    print(calc_seg_meas(gt_sequence, output_sequence))


if __name__ == '__main__':
    seg_measure_unit_test()
