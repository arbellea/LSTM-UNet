import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from scipy.ndimage import grey_dilation
import argparse


def main(root_dir, seq, raw_file_template, seg_file_template, tra_file_template=None, force=False, FOV=80):
    if os.path.exists(os.path.join(root_dir, 'metadata_{}.pickle'.format(seq))) and not force:
        raise FileExistsError('File {} already exists! use --force option to overwrite')

    all_image_files = os.listdir(os.path.join(root_dir, os.path.join(os.path.dirname(raw_file_template))))
    all_seg_files = os.listdir(os.path.join(root_dir, os.path.join(os.path.dirname(seg_file_template))))
    all_tra_files = os.listdir(os.path.join(root_dir, os.path.join(os.path.dirname(tra_file_template))))
    t = 0
    seq_metadata = {'filelist': [], 'max': 0, 'min': np.inf, 'shape': None}
    valid_seg = None
    while True:
        if os.path.basename(raw_file_template.format(t)) in all_image_files:

            im_fname = raw_file_template.format(t)
            seg_fname = seg_file_template.format(t)
            tra_fname = tra_file_template.format(t) if tra_file_template is not None else None

            im = cv2.imread(os.path.join(root_dir, im_fname), -1)
            seq_metadata['max'] = np.maximum(seq_metadata['max'], im.max())
            seq_metadata['min'] = np.minimum(seq_metadata['min'], im.min())
            if seq_metadata['shape'] is None:
                seq_metadata['shape'] = im.shape
            elif not np.all(seq_metadata['shape'] == im.shape):
                raise ValueError(
                    'Image shape should be consistent for full sequence, expected {}, got {} for image {}'.format(
                        seq_metadata['shape'], im.shape, im_fname))

            if os.path.basename(seg_file_template.format(t)) in all_seg_files:
                seg = cv2.imread(os.path.join(root_dir, seg_fname), -1)
                if tra_fname and os.path.basename(tra_fname) in all_tra_files:
                    tra = cv2.imread(os.path.join(root_dir, tra_fname), -1)
                    if FOV:
                        tra[-FOV:,:]=0
                        tra[:FOV,:]=0
                        tra[:FOV]=0
                        tra[:,-FOV:]=0
                    if len(np.unique(seg))>=len(np.unique(tra)):
                        valid_seg = 'y'
                        print('y ', tra_fname)
                    else:
                        valid_seg = 'n'
                        print('n ', tra_fname)
                elif valid_seg == 'yes to all':
                    pass
                else:
                    im = (im - im.min()) / (im.max() - im.min())
                    imR = im.copy()
                    imG = im.copy()
                    imB = im.copy()
                    strel = np.zeros((5, 5))
                    dilation = grey_dilation(seg.astype(np.int32), structure=strel.astype(np.int8))
                    seg_boundary = np.zeros_like(im, dtype=np.bool)
                    seg_boundary[np.logical_and(np.not_equal(seg, dilation), np.greater(dilation, 0))] = True

                    imR[seg_boundary] = 1
                    imG[seg_boundary] = 0
                    imB[seg_boundary] = 0
                    imrgb = np.stack([imR, imG, imB], 2)
                    plt.figure(1)
                    plt.cla()
                    plt.imshow(imrgb)
                    plt.title('T = {}'.format(t))
                    plt.pause(0.1)
                    valid_seg = input('Frame {}: Are all cells in the frame annotated [Y/n/yes to all]? '.format(t)).lower()

                row = (im_fname, seg_fname, tra_fname, valid_seg in ['', 'y', 'yes', 'yes to all'])
            else:
                row = (im_fname, None, tra_fname, None)
            seq_metadata['filelist'].append(row)

        else:
            break
        t += 1
    with open(os.path.join(root_dir, 'metadata_{}.pickle'.format(seq)), 'wb') as f:
        pickle.dump(seq_metadata, f, pickle.HIGHEST_PROTOCOL)


def get_default_run():
    root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training'
    dataset = 'DIC-C2DH-HeLa'
    seq = '01'
    # seq_metadata[filelist] will be a list of tuples holding (raw_fname, seg_fname, tra_filename, is_seg_val)
    raw_file_template = os.path.join(seq, 't{:03d}.tif')
    seg_file_template = os.path.join('{}_GT'.format(seq), 'SEG', 'man_seg{:03d}.tif')
    tra_file_template = os.path.join('{}_GT'.format(seq), 'TRA',
                                     'man_track{:03d}.tif')  # Optional

    root_dir = os.path.join(root_dir, dataset)

    return root_dir, seq, raw_file_template, seg_file_template, tra_file_template


if __name__ == '__main__':

    root_dir_, seq_, raw_file_template_, seg_file_template_, tra_file_template_ = get_default_run()

    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('--root_dir', dest='root_dir', type=str,
                            help="Root directory of sequence, example: '~/CellTrackingChallenge/Train/Fluo-N2DH-SIM+")
    arg_parser.add_argument('--seq', dest='seq', type=str,
                            help="Sequence number (two digit) , example: '01' or '02' ")
    arg_parser.add_argument('--raw_file_template', dest='raw_file_template', type=str,
                            help="Template for image sequences, example: '01/t{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")
    arg_parser.add_argument('--seg_file_template', dest='seg_file_template', type=str,
                            help="Template for image sequences segmentation , example: '01_GT/SEG/man_seg{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")
    arg_parser.add_argument('--tra_file_template', dest='tra_file_template', type=str,
                            help="Optional!. Template for image sequences tracking lables , example: '01_GT/TRA/man_track{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")

    arg_parser.add_argument('--force', dest='force', help='Force overwrite existing metadata pickle')
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    if len(sys_args) > 1:

        if sys_args < 5:
            raise SyntaxError('Please input all parameters: root_dir, raw_file_template, seg_file_'
                              'template and optionaly tra_file_template ')
        root_dir_ = input_args.root_dir
        raw_file_template_ = input_args.raw_file_template
        seg_file_template_ = input_args.seg_file_template
        tra_file_template_ = input_args.tra_file_template
    input_args.force = True
    main(root_dir_, seq_, raw_file_template_, seg_file_template_, tra_file_template_, input_args.force)
