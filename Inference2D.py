import argparse
import os
import pickle
import scipy.ndimage
import cv2
import numpy as np
import tensorflow as tf
import Networks as Nets
from Params import CTCInferenceParams
from distutils.util import strtobool
import DataHandeling
import sys
from utils import log_print, get_model, bbox_crop, bbox_fill

__author__ = 'arbellea@post.bgu.ac.il'

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')


def inference():
    # Load Model
    with open(os.path.join(params.model_path, 'model_params.pickle'), 'rb') as fobj:
        model_dict = pickle.load(fobj)
    model_cls = get_model(model_dict['name'])

    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        model = model_cls(*model_dict['params'], data_format=params.data_format, pad_image=True)
        model.load_weights(os.path.join(params.model_path, 'model.ckpt'))
        log_print("Restored from {}".format(os.path.join(params.model_path, 'model.ckpt')))
    base_out_temp_vis_fname = base_out_temp_label_fname = base_out_fname = None
    if not params.dry_run:
        if params.save_intermediate_path:
            base_out_temp_vis_fname = os.path.join(params.save_intermediate_vis_path, 'softmax{time:03d}.tif')
            base_out_temp_label_fname = os.path.join(params.save_intermediate_label_path, 'mask{time:03d}.tif')
        base_out_fname = os.path.join(params.output_path, 'mask{time:03d}.tif')
    dataset = params.data_reader(params.sequence_path, params.filename_format,
                                 pre_sequence_frames=params.pre_sequence_frames).dataset
    try:
        for T, image in enumerate(dataset):
            t = T - params.pre_sequence_frames

            image_shape = image.shape
            if len(image_shape) == 2:
                if params.data_format == 'NCHW':
                    image = tf.reshape(image, [1, 1, 1, image_shape[0], image_shape[1]])
                else:
                    image = tf.reshape(image, [1, 1, image_shape[0], image_shape[1], 1])
            elif len(image_shape) == 3:
                image = tf.reshape(image, [1, 1, image_shape[0], image_shape[1], image_shape[2]])
            else:
                raise ValueError()

            _, image_softmax = model(image, training=False)
            image_softmax_np = np.squeeze(image_softmax.numpy(), (0, 1))
            if t < 0:
                continue

            if not params.dry_run:

                seg_edge = np.greater_equal(image_softmax_np[2], 0.2)
                seg_cell = np.logical_and(np.equal(np.argmax(image_softmax_np, 0), 1).astype(np.float32),
                                          np.logical_not(seg_edge))
                seg_edge = seg_edge.astype(np.float32)
                seg_cell = scipy.ndimage.morphology.binary_fill_holes(seg_cell).astype(np.float32)
                seg_edge = np.maximum((seg_edge - seg_cell), 0)
                cc_out = cv2.connectedComponentsWithStats(seg_cell.astype(np.uint8), 8, cv2.CV_32S)
                num_cells = cc_out[0]
                labels = cc_out[1]
                stats = cc_out[2]

                dist, ind = scipy.ndimage.morphology.distance_transform_edt(1 - seg_cell, return_indices=True)
                labels = labels[ind[0, :], ind[1, :]] * seg_edge * (dist < params.edge_dist) + labels

                for n in range(1, num_cells):
                    bw = labels == n
                    if not np.any(bw):
                        continue

                    bw_crop, loc = bbox_crop(bw)

                    fill_crop = scipy.ndimage.morphology.binary_fill_holes(bw_crop).astype(np.float32)
                    fill_diff = fill_crop - bw_crop
                    bw_fill = bbox_fill(bw, fill_diff, loc)
                    labels = labels + bw_fill * n

                # filter by fov
                if params.FOV:
                    fov_im = np.ones_like(labels)
                    fov_im[:params.FOV, :] = 0
                    fov_im[-params.FOV:, :] = 0
                    fov_im[:, params.FOV] = 0
                    fov_im[:, -params.FOV:] = 0
                    fov_labels = labels * fov_im
                    unique_fov_labels = np.unique(fov_labels.flatten())
                    remove_ind = np.setdiff1d(np.arange(num_cells), unique_fov_labels)
                else:
                    remove_ind = []
                if params.save_intermediate:

                    if params.data_format == 'NCHW':
                        image_softmax_np = np.transpose(image_softmax_np, (1, 2, 0))
                    out_fname = base_out_temp_vis_fname.format(time=t)
                    sigoutnp_vis = np.flip(np.round(image_softmax_np * (2 ** 16 - 1)).astype(np.uint16), 2)
                    cv2.imwrite(filename=out_fname, img=sigoutnp_vis.astype(np.uint16))
                    log_print("Saved File: {}".format(out_fname))

                labels_out = np.zeros_like(labels, dtype=np.uint16)
                # isbi_out_dict = {}
                p = 0
                for n in range(1, num_cells):
                    area = stats[n, cv2.CC_STAT_AREA]
                    if params.min_cell_size <= area <= params.max_cell_size and not (n in remove_ind):
                        p += 1
                        # isbi_out_dict[p] = [p, 0, 0, 0]
                        labels_out[labels == n] = p

                    else:
                        labels[labels == n] = 0
                out_fname = base_out_fname.format(time=t)
                cv2.imwrite(filename=out_fname, img=labels_out.astype(np.uint16))
                log_print("Saved File: {}".format(out_fname))
                if params.save_intermediate:
                    out_fname = base_out_temp_label_fname.format(time=t)
                    cv2.imwrite(filename=out_fname, img=labels_out.astype(np.uint16))
                    log_print("Saved File: {}".format(out_fname))

    except (KeyboardInterrupt, ValueError) as err:
        print('Error: {}'.format(str(err)))

    finally:
        print('Done!')


if __name__ == '__main__':

    class AddNets(argparse.Action):
        import Networks as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            setattr(namespace, self.dest, nets)


    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataHandeling, values)
            setattr(namespace, self.dest, reader)


    class AddDatasets(argparse.Action):

        def __init__(self, option_strings, dest, *args, **kwargs):

            super(AddDatasets, self).__init__(option_strings, dest, *args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):

            if len(values) % 2:
                raise ValueError("dataset values should be of length 2*N where N is the number of datasets")
            datastets = []
            for i in range(0, len(values), 2):
                datastets.append((values[i], strtobool(values[i + 1])))
            setattr(namespace, self.dest, datastets)


    arg_parser = argparse.ArgumentParser(description='Run Inference LSTMUnet Segmentation')
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                            help="Visible GPUs: example, '0,2,3', use -1 for CPU")
    arg_parser.add_argument('--model_path', dest='model_path', type=str,
                            help="Path to trained model generated by train2D.py, folder should contain model.ckpt.*")

    arg_parser.add_argument('--sequence_path', dest='sequence_path', type=str,
                            help="Path to sequence images. Folder should contain image files")
    arg_parser.add_argument('--filename_format', dest='filename_format', type=str,
                            help="Format of file using wildcard (*) to indicate timestep. Default: 't*.tif'")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")

    arg_parser.add_argument('--min_cell_size', dest='min_cell_size', type=int,
                            help="Minimum cell size")
    arg_parser.add_argument('--max_cell_size', dest='max_cell_size', type=int,
                            help="Maximum cell size")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--edge_dist', dest='edge_dist', type=int,
                            help="Maximum edge width to add to cell object")
    arg_parser.add_argument('--pre_sequence_frames', dest='pre_sequence_frames', type=int,
                            help="Number of frames to run before sequence, uses mirror of first N frames.")
    arg_parser.add_argument('--save_intermediate', dest='save_intermediate', action='store_const', const=True,
                            help="Save intermediate files")
    arg_parser.add_argument('--save_intermediate_path', dest='save_intermediate_path', type=str,
                            help="Path to save intermediate files, used only with --save_intermediate")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    params = CTCInferenceParams(args_dict)
    tf_eps = tf.constant(1E-8, name='epsilon')
    try:
        inference()
    finally:
        log_print('Done')
