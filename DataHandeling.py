import random
import tensorflow as tf
import os
import glob
import cv2
import queue
import threading
import numpy as np
import pickle
import utils
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import grey_dilation, affine_transform  # measurements, grey_erosion

__author__ = 'assafarbelle'


class CTCRAMReaderSequence2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', randomize=True, return_dist=False, keep_sample=1, elastic_augmentation=True):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.coord = None
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = return_dist
        self.num_threads = num_threads
        self.keep_sample = keep_sample

        self.q_list, self.q_stat_list = self._create_queues()
        np.random.seed(1)

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'

        sequence_folder_list = [(os.path.join(root_dir, 'Fluo-N2DH-SIM+'), '01'),
                                (os.path.join(root_dir, 'Fluo-N2DH-SIM+'), '02')]
        image_crop_size = (128, 128)
        unroll_len = 7
        deal_with_end = 0
        batch_size = 4
        queue_capacity = 250
        num_threads = 2
        data_format = 'NCHW'
        randomize = True
        return_dist = False
        keep_sample = 1
        elastic_augmentation = True
        data = cls(sequence_folder_list, image_crop_size, unroll_len, deal_with_end, batch_size, queue_capacity,
                   num_threads, data_format, randomize, return_dist, keep_sample, elastic_augmentation)

        debug = True
        data.start_queues(debug=debug)
        for i in range(100):
            image_batch, seg_batch, full_seg, is_last, fnames = data.get_batch()
            utils.log_print(image_batch.shape, seg_batch.shape, is_last.shape)

    def _read_sequence_to_ram_(self):
        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = None
            sequence_folder_orig = sequence_folder
            if isinstance(sequence_folder, tuple):
                if len(sequence_folder) == 2:
                    sequence_folder, seq = sequence_folder
                elif len(sequence_folder) == 3:
                    sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}: {}'.format(sequence_folder, seq))
            with open(os.path.join(sequence_folder, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                metadata = pickle.load(fobj)

            filename_list = metadata['filelist']
            img_size = metadata['shape']
            if len(img_size) == 3:
                img_size = img_size[1:]
            all_images = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_seg = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_full_seg = np.zeros((len(filename_list)))
            keep_rate = self.keep_sample
            original_size = 0
            downampled_size = 0
            for t, filename in enumerate(filename_list):
                img = cv2.imread(os.path.join(sequence_folder, filename[0]), -1)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(sequence_folder, filename[0])))
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std())
                full_seg = 1 if filename[3] is True else 0
                if full_seg == 1:
                    original_size += 1

                keep_seg = (np.random.rand() < keep_rate) and train_set

                full_seg = full_seg if keep_seg else 0
                if full_seg == 1:
                    downampled_size += 1

                if filename[1] is None or not keep_seg:
                    seg = np.ones(img.shape[:2]) * (-1)
                elif not full_seg:
                    seg = cv2.imread(os.path.join(sequence_folder, filename[1]), -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                        full_seg = -1

                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                else:
                    seg = cv2.imread(os.path.join(sequence_folder, filename[1]), -1)
                all_images[t] = img
                all_seg[t] = seg
                all_full_seg[t] = full_seg
            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))

            self.sequence_data[sequence_folder_orig] = {'images': np.array(all_images), 'segs': np.array(all_seg),
                                                        'full_seg': all_full_seg, 'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        # if isinstance(sequence_folder, tuple):
        #     sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape
        if seg:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=-1, flags=cv2.INTER_NEAREST)
            trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
        else:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)

        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):
        dxr = random_state.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    @staticmethod
    def _adjust_brightness_(image, delta):
        """
        Args:
        image (numpy.ndarray)
        delta
        """

        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        """
        Args:
        image (numpy.ndarray)
        factor
        """

        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def _load_and_enqueue(self, q, q_stat):
        dist_crop = None
        unroll_len = self.unroll_len
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                img_size = seq_data['metadata']['shape']
                random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
                random_reverse = np.random.randint(0, 2) if self.randomize else 0
                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_x = 0

                flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if self.elastic_augmentation:
                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size,
                                                                                   self.sub_seq_size[1] * 0.08)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size, self.sub_seq_size[1] * 2,
                                                                  self.sub_seq_size[1] * 0.15,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['metadata']['filelist'])))

                if random_reverse:
                    filename_idx.reverse()
                if random_sub_sample:
                    filename_idx = filename_idx[::random_sub_sample]
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_y_stop = crop_y + self.sub_seq_size[0]
                crop_x_stop = crop_x + self.sub_seq_size[1]
                img_crops = seq_data['images'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                img_max = seq_data['images'].max()

                seg_crops = seq_data['segs'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                processed_image = []
                processed_seg = []
                full_seg_seq = []
                processed_dist = []
                is_last = []
                all_fnames = []
                for t, file_idx in enumerate(filename_idx):
                    all_times = [time.time()]
                    filename = seq_data['metadata']['filelist'][file_idx][0]
                    img_crop = img_crops[file_idx].copy()
                    seg_crop = seg_crops[file_idx].copy()
                    full_seg = seq_data['full_seg'][file_idx]
                    if self.randomize:
                        # contrast factor between [0.5, 1.5]
                        random_constrast_factor = np.random.rand() + 0.5
                        # random brightness delta plus/minus 10% of maximum value
                        random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                        img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                        img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)

                    if self.elastic_augmentation:
                        trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                        img_crop = trans_img
                        if np.any(np.isnan(img_crop)):
                            raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if np.any(np.isinf(img_crop)):
                            raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if not np.equal(seg_crop, -1).all():
                            seg_not_valid = np.equal(seg_crop, -1)
                            labeled_gt = seg_crop
                            labeled_gt[:, 0] = 0
                            labeled_gt[:, -1] = 0
                            labeled_gt[-1, :] = 0
                            labeled_gt[0, :] = 0
                            trans_seg = self._get_transformed_image_(labeled_gt.astype(np.float32), affine_matrix,
                                                                     indices, seg=True)
                            trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                                                                           affine_matrix,
                                                                           indices, seg=True)
                            trans_seg_fix = self._fix_transformed_segmentation(trans_seg)
                            trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                            seg_crop = trans_seg_fix
                            seg_crop[trans_not_valid] = -1
                            if np.any(np.isnan(seg_crop)):
                                raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                            if np.any(np.isinf(seg_crop)):
                                raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                    else:
                        seg_crop = self._fix_transformed_segmentation(seg_crop)
                    if flip[0]:
                        img_crop = cv2.flip(img_crop, 0)
                        seg_crop = cv2.flip(seg_crop, 0)
                    if flip[1]:
                        img_crop = cv2.flip(img_crop, 1)
                        seg_crop = cv2.flip(seg_crop, 1)
                    if rotate > 0:
                        img_crop = np.rot90(img_crop, rotate)
                        seg_crop = np.rot90(seg_crop, rotate)
                    if self.return_dist:
                        if full_seg == -1:
                            dist_crop = np.zeros(self.dist_sub_seq_size)
                        else:
                            dist_crop, _ = self._gt2dist_(seg_crop)
                        all_times.append(time.time())
                        processed_dist.append(dist_crop)
                    is_last_frame = 1. if (t + 1) < len(filename_idx) else 0.
                    if self.num_threads == 1:
                        try:

                            while q_stat().numpy() > 0.9:
                                if self.coord.should_stop():
                                    return
                                time.sleep(1)

                            if self.return_dist:
                                q.enqueue(
                                    [img_crop, seg_crop, max(0, full_seg), is_last_frame, filename, dist_crop])
                            else:
                                q.enqueue([img_crop, seg_crop, max(0, full_seg), is_last_frame, filename])

                        except tf.errors.CancelledError:
                            pass
                    else:
                        is_last.append(is_last_frame)
                        processed_image.append(img_crop)
                        processed_seg.append(seg_crop)
                        all_fnames.append(filename)
                        full_seg_seq.append(max(0, full_seg))
                        if self.coord.should_stop():
                            return
                if self.num_threads == 1:
                    continue

                try:

                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(1)

                    if self.return_dist:
                        q.enqueue_many(
                            [processed_image, processed_seg, full_seg_seq, is_last, all_fnames, processed_dist])
                    else:
                        q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last, all_fnames])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            self.coord.request_stop()
            raise err

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size, (), (), ()]
            if self.return_dist:
                dtypes += [tf.float32]
                shapes += [self.dist_sub_seq_size]
            q_list = []

            q_stat_list = []
            for b in range(self.batch_size):
                q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q_{}'.format(b))
                q_list.append(q)
                q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            dist_list = []
            full_seg_list = []
            is_last_list = []
            # fname_list = []
            for q in self.q_list:
                if self.return_dist:
                    img, seg, full_seg, is_last, fnames, dist = q.dequeue_many(self.unroll_len)
                    dist_list.append(dist)
                else:
                    img, seg, full_seg, is_last, fnames = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])
                # fname_list.append(fnames)

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            # fnames_batch = tf.stack(fname_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            dist_batch = tf.stack(dist_list, axis=0) if self.return_dist else None

            if self.data_format == 'NHWC':
                image_batch = tf.expand_dims(image_batch, 4)
                seg_batch = tf.expand_dims(seg_batch, 4)
            elif self.data_format == 'NCHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
            else:
                raise ValueError()

        if self.return_dist:
            return image_batch, seg_batch, full_seg_batch, is_last_batch, dist_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch

    def _create_sequence_queue(self):
        sequence_queue = queue.Queue(maxsize=len(self.sequence_folder_list))
        for sequence in self.sequence_folder_list:
            sequence_queue.put(sequence)
        return sequence_queue

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


class LSCRAMReader2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', randomize=True, return_dist=False, keep_sample=1, elastic_augmentation=True):
        self.coord = None
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = return_dist
        self.num_threads = num_threads
        self.keep_sample = keep_sample

        self.q_list, self.q_stat_list = self._create_queues()
        np.random.seed(1)

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/LeafSegmentation/CVPPP2014_LSC_training_data'

        folder_list = [os.path.join(root_dir, 'A1'),
                       os.path.join(root_dir, 'A2'),
                       os.path.join(root_dir, 'A3')]
        image_crop_size = (128, 128)
        unroll_len = 7
        deal_with_end = 0
        batch_size = 4
        queue_capacity = 250
        num_threads = 2
        data_format = 'NCHW'
        randomize = True
        return_dist = False
        keep_sample = 1
        elastic_augmentation = True
        data = cls(folder_list, image_crop_size, unroll_len, deal_with_end, batch_size, queue_capacity,
                   num_threads, data_format, randomize, return_dist, keep_sample, elastic_augmentation)

        debug = True
        data.start_queues(debug=debug)
        for i in range(100):
            image_batch, seg_batch, full_seg, is_last, fnames = data.get_batch()
            utils.log_print(image_batch.shape, seg_batch.shape, is_last.shape)

    def _read_sequence_to_ram_(self):
        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = None
            if isinstance(sequence_folder, tuple):
                if len(sequence_folder) == 2:
                    sequence_folder, seq = sequence_folder
                elif len(sequence_folder) == 3:
                    sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}: {}'.format(sequence_folder, seq))
            filename_list = glob.glob(os.path.join(sequence_folder, 'plant*_label.png'))
            metadata = {'filelist': filename_list, 'max_value': 0.}
            all_images = []  # = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_seg = []  # np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_full_seg = []  # np.zeros((len(filename_list)))
            keep_rate = 1
            original_size = 0
            downampled_size = 0
            for t, filename in enumerate(filename_list):
                filename = os.path.basename(filename)
                rgb_filename = filename.replace('label', 'rgb')
                img = cv2.cvtColor(cv2.imread(os.path.join(sequence_folder, rgb_filename)),
                                   cv2.COLOR_BGRA2RGB)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(sequence_folder, rgb_filename)))
                if 'shape' not in metadata.keys():
                    metadata['shape'] = img.shape
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std())
                full_seg = 1
                if full_seg == 1:
                    original_size += 1

                keep_seg = (np.random.rand() < keep_rate) and train_set

                full_seg = full_seg if keep_seg else 0
                if full_seg == 1:
                    downampled_size += 1

                if filename is None or not keep_seg:
                    seg = np.ones(img.shape[:2]) * (-1)
                elif not full_seg:
                    seg = cv2.imread(os.path.join(sequence_folder, filename), -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                        full_seg = -1

                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                else:
                    seg = cv2.imread(os.path.join(sequence_folder, filename), cv2.IMREAD_GRAYSCALE)
                all_images.append(img)
                metadata['max_value'] = np.maximum(metadata['max_value'], img.max())
                all_seg.append(seg)
                all_full_seg.append(full_seg)
            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))

            self.sequence_data[sequence_folder] = {'images': all_images, 'segs': all_seg,
                                                   'full_seg': all_full_seg, 'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape[:2]
        if seg:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=-1, flags=cv2.INTER_NEAREST)
            trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
        else:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)

        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):
        dxr = random_state.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    @staticmethod
    def _adjust_brightness_(image, delta):
        """
        Args:
        image (numpy.ndarray)
        delta
        """

        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        """
        Args:
        image (numpy.ndarray)
        factor
        """

        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def _load_and_enqueue(self, q, q_stat):

        unroll_len = self.unroll_len
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                img_size = seq_data['metadata']['shape']

                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_x = 0

                flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if self.elastic_augmentation:
                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size,
                                                                                   self.sub_seq_size[1] * 0.08)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size, self.sub_seq_size[1] * 2,
                                                                  self.sub_seq_size[1] * 0.15,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['metadata']['filelist'])))
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_y_stop = crop_y + self.sub_seq_size[0]
                crop_x_stop = crop_x + self.sub_seq_size[1]
                file_idx = random.choice(filename_idx)
                img_crops = seq_data['images'][file_idx][crop_y:crop_y_stop, crop_x:crop_x_stop]
                img_max = seq_data['images'][file_idx].max()

                seg_crops = seq_data['segs'][file_idx][crop_y:crop_y_stop, crop_x:crop_x_stop]
                processed_image = []
                processed_seg = []
                full_seg_seq = []
                processed_dist = []
                is_last = []
                all_fnames = []
                all_times = [time.time()]
                filename = seq_data['metadata']['filelist'][file_idx]
                img_crop = img_crops.copy()
                seg_crop = seg_crops.copy()
                full_seg = seq_data['full_seg'][file_idx]
                if self.randomize:
                    # contrast factor between [0.5, 1.5]
                    random_constrast_factor = np.random.rand() + 0.5
                    # random brightness delta plus/minus 10% of maximum value
                    random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                    img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                    img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)

                if self.elastic_augmentation:
                    trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                    img_crop = trans_img
                    if np.any(np.isnan(img_crop)):
                        raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                    if np.any(np.isinf(img_crop)):
                        raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                    if not np.equal(seg_crop, -1).all():
                        seg_not_valid = np.equal(seg_crop, -1)
                        labeled_gt = seg_crop
                        labeled_gt[:, 0] = 0
                        labeled_gt[:, -1] = 0
                        labeled_gt[-1, :] = 0
                        labeled_gt[0, :] = 0
                        trans_seg = self._get_transformed_image_(labeled_gt.astype(np.float32), affine_matrix,
                                                                 indices, seg=True)
                        trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                                                                       affine_matrix,
                                                                       indices, seg=True)
                        trans_seg_fix = self._fix_transformed_segmentation(trans_seg)
                        trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                        seg_crop = trans_seg_fix
                        seg_crop[trans_not_valid] = -1
                        if np.any(np.isnan(seg_crop)):
                            raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                        if np.any(np.isinf(seg_crop)):
                            raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                else:
                    seg_crop = self._fix_transformed_segmentation(seg_crop)
                if flip[0]:
                    img_crop = cv2.flip(img_crop, 0)
                    seg_crop = cv2.flip(seg_crop, 0)
                if flip[1]:
                    img_crop = cv2.flip(img_crop, 1)
                    seg_crop = cv2.flip(seg_crop, 1)
                if rotate > 0:
                    img_crop = np.rot90(img_crop, rotate)
                    seg_crop = np.rot90(seg_crop, rotate)
                if self.return_dist:
                    if full_seg == -1:
                        dist_crop = np.zeros(self.dist_sub_seq_size)
                    else:
                        dist_crop, _ = self._gt2dist_(seg_crop)
                    all_times.append(time.time())
                    processed_dist.append(dist_crop)
                is_last_frame = 1.
                is_last.append(is_last_frame)
                processed_image.append(img_crop)
                processed_seg.append(seg_crop)
                all_fnames.append(filename)
                full_seg_seq.append(max(0, full_seg))
                if self.coord.should_stop():
                    return

                try:
                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(1)

                    if self.return_dist:
                        q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last, all_fnames,
                                        processed_dist])
                    else:
                        q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last, all_fnames])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            self.coord.request_stop()
            raise err

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size, (), (), ()]
            if self.return_dist:
                dtypes += [tf.float32]
                shapes += [self.dist_sub_seq_size]
            q_list = []

            q_stat_list = []
            for b in range(self.batch_size):
                q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q_{}'.format(b))
                q_list.append(q)
                q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            dist_list = []
            full_seg_list = []
            is_last_list = []
            fname_list = []
            for q in self.q_list:
                if self.return_dist:
                    img, seg, full_seg, is_last, fnames, dist = q.dequeue_many(self.unroll_len)
                    dist_list.append(dist)
                else:
                    img, seg, full_seg, is_last, fnames = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])
                fname_list.append(fnames)

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            fnames_batch = tf.stack(fname_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            dist_batch = tf.stack(dist_list, axis=0) if self.return_dist else None

            if self.data_format == 'NHWC':
                image_batch = tf.expand_dims(image_batch, 4)
                seg_batch = tf.expand_dims(seg_batch, 4)
            elif self.data_format == 'NCHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
            else:
                raise ValueError()

        if self.return_dist:
            return image_batch, seg_batch, full_seg_batch, is_last_batch, dist_batch, fnames_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch, fnames_batch

    def _create_sequence_queue(self):
        sequence_queue = queue.Queue(maxsize=len(self.sequence_folder_list))
        for sequence in self.sequence_folder_list:
            sequence_queue.put(sequence)
        return sequence_queue

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


class CTCSegReaderSequence3D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(32, 128, 128), unroll_len=7, deal_with_end=0,
                 batch_size=4, queue_capacity=32, num_threads=3, data_format='NCDHW', randomize=True, return_dist=False,
                 keep_sample=1, elastic_augmentation=False, switch_to_local_db=False,
                 local_db_replace=('/persistent', '/data3d')):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = return_dist
        self.num_threads = num_threads
        self.keep_sample = keep_sample
        self.switch_to_local_db = switch_to_local_db
        self.local_db_replace = local_db_replace

        self.q_list, self.q_stat_list = self._create_queues()
        self.coord = None

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'
        dataset = 'Fluo-C3DL-MDA231'
        sequence = '01'
        train_data = True
        image_crop_size = (16, 80, 80)
        unroll_len = 3
        num_threads = 3
        batch_size = 4
        debug = True
        queue_capacity = 200
        sequence_folder_list = [(os.path.join(root_dir, dataset), sequence, train_data)]
        data = cls(sequence_folder_list, batch_size=batch_size, image_crop_size=image_crop_size, unroll_len=unroll_len,
                   elastic_augmentation=True, num_threads=num_threads, queue_capacity=queue_capacity)
        coord = tf.train.Coordinator()
        data.start_queues(coord, debug=debug)
        time.sleep(30)
        for i in range(100):
            time.sleep(0.1)
            start = time.time()
            image_batch_list, seg_batch_list, full_seg_batch_list, is_last_batch = data.get_batch()
            print([qs().numpy() for qs in data.q_stat_list])
            print('Time: {}'.format(time.time() - start))
            # print([q.size().numpy() for q in data.q_list])
            print(image_batch_list[0].shape)

    def _read_sequence_to_ram_(self):

        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = '01'
            sequence_folder_orig = sequence_folder
            if isinstance(sequence_folder, tuple):
                sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}'.format(sequence_folder))
            with open(os.path.join(sequence_folder, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                metadata = pickle.load(fobj)
            filename_list = metadata['filelist']
            original_size = 0
            keep_inds = []
            for t, f in enumerate(filename_list):
                if f[1] is not None and f[2] is True:
                    original_size += 1
                    keep_inds.append(t)
                elif isinstance(f[1], list):
                    for z, (slice_f, valid_f) in enumerate(zip(f[1], f[2])):
                        if slice_f is not None and valid_f is True:
                            original_size += 1
                            keep_inds.append(t)
                            break
            keep_inds = np.array(keep_inds)

            keep_rate = self.keep_sample if train_set else 0
            keep_num = np.round(keep_rate * original_size).astype(np.int16)
            keep_vec = np.zeros(len(filename_list))
            if keep_num > 0 and len(keep_inds) > 0:
                keep_inds = np.random.choice(keep_inds, keep_num, replace=False)
                keep_vec[keep_inds] = 1
            downampled_size = keep_num

            for t, (filename, keep_seg) in enumerate(zip(filename_list, keep_vec)):

                if not keep_seg:
                    filename[1] = None
                    filename[2] = None

            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))

            self.sequence_data[sequence_folder_orig] = {'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        # if isinstance(sequence_folder, tuple):
        #     sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _points2affine3d_(pts1, pts2):
        """
         pts need to be a set of 4 point of shape 4x3 for zyx
        :param pts1:
        :param pts2:
        :return:
        """
        x_mat = np.zeros((12, 12))
        for p_ind, pt in enumerate(pts1):
            pt1 = np.concatenate((pt, np.array([1])), axis=0)
            x_mat[p_ind * 3] = np.concatenate((pt1, np.zeros(8)), axis=0)
            x_mat[p_ind * 3 + 1] = np.concatenate((np.zeros(4), pt1, np.zeros(4)), axis=0)
            x_mat[p_ind * 3 + 2] = np.concatenate((np.zeros(8), pt1), axis=0)

        inv_x = np.linalg.inv(x_mat.astype(np.float32))
        pts2 = np.reshape(pts2, -1)

        a = np.dot(inv_x, pts2)
        a_mat = np.concatenate((a, np.array([0, 0, 0, 1])), axis=0)
        a_mat = np.reshape(a_mat, (4, 4))
        return a_mat

    def _get_elastic_affine_matrix_(self, shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        depth, height, width = shape_size
        pts1 = np.float32([[0, 0, 0], [0, height, width], [depth, height, 0], [depth, 0, width]])
        pts2 = pts1 + random_state.uniform(size=pts1.shape).astype(np.float32) * (alpha_affine * np.array(shape_size))
        affine_matrix = self._points2affine3d_(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape
        try:
            if seg:
                trans_img = affine_transform(image, affine_matrix, order=0, mode='constant', cval=-1)
                trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
            else:
                trans_img = affine_transform(image, affine_matrix, order=1, mode='reflect')
                trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)
        except RuntimeError:
            a = affine_matrix[:3, :3]
            b = affine_matrix[:3, 3]
            if seg:
                trans_img = affine_transform(image, a, offset=b, order=0, mode='constant', cval=-1)
                trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
            else:
                trans_img = affine_transform(image, a, offset=b, order=1, mode='reflect')
                trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)
        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):

        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha[0]
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha[1]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha[2]

        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg, edge_z):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3, 3)) if edge_z else np.zeros((1, 3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2
        return bw

    @staticmethod
    def _label_class_segmentation(trans_seg, labels):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        trans_seg = np.round(trans_seg)
        bw = np.zeros_like(trans_seg)
        if len(np.unique(trans_seg)) == 0:
            return np.minimum(trans_seg, 1)
        for this_label in labels:
            if this_label == 0 or (this_label not in trans_seg):
                continue

            label_bw = np.equal(trans_seg, this_label)
            # cc = cv2.connectedComponentsWithStats(label_bw.astype(np.uint8), 8, cv2.CV_32S)
            # num = cc[0]
            # label_bw_l = cc[1]
            # stats = cc[2]
            # ll = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            # label_bw = np.equal(label_bw_l, ll)
            label_bw = binary_dilation(label_bw.astype(np.uint8), np.ones((3, 3, 3)))
            bw_dilate = cv2.dilate(label_bw.astype(np.uint8), np.ones((3, 3)))
            edge = np.logical_xor(label_bw, bw_dilate)
            bw[label_bw.astype(np.bool)] = 1
            bw[edge] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    def read_images(self, sequence_folder, file_idx, start_z=None, stop_z=None):
        sequence_folder_orig = sequence_folder
        if isinstance(sequence_folder, tuple):
            sequence_folder, seq, train_set = sequence_folder
        if self.switch_to_local_db and os.path.exists(os.path.join(self.local_db_replace[1], '.done_data_move')):
            this_dir = sequence_folder.replace(self.local_db_replace[0], self.local_db_replace[1])
        else:
            this_dir = sequence_folder
        metadata = self.sequence_data[sequence_folder_orig]['metadata']
        filename_list = metadata['filelist']
        filename = filename_list[file_idx]
        img = utils.read_multi_tiff(os.path.join(this_dir, filename[0]), start_z=start_z, stop_z=stop_z)
        if img is None:
            print('Could Not Load Image: {}'.format(filename[0]))

        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std())
        seg = None
        full_seg = 1 if filename[2] is True else 0

        if filename[1] is None:
            seg = np.ones(img.shape[:3]) * (-1)
        elif not full_seg:
            if isinstance(filename[1], str):
                seg = utils.read_multi_tiff(os.path.join(this_dir, filename[1]), start_z=start_z, stop_z=stop_z)
                seg = seg.astype(np.float32)
            elif isinstance(filename[1], list):
                seg = np.ones(img.shape[:3]) * (-1)
                for slice_name, valid in zip(filename[1], filename[2]):
                    z = int(slice_name[-7:-4]) - start_z
                    if z < 0 or z >= seg.shape[0]:
                        continue
                    this_slice = cv2.imread(os.path.join(this_dir, slice_name), -1)
                    this_slice = this_slice.astype(np.float32)
                    if not valid or 'DRO' in sequence_folder:
                        this_slice[this_slice == 0] = -1
                    seg[z] = this_slice
                seg = seg

        else:
            seg = utils.read_multi_tiff(os.path.join(this_dir, filename[1]), start_z=start_z, stop_z=stop_z)

        return img, seg, full_seg

    def _load_and_enqueue(self, q, q_stat):

        unroll_len = self.unroll_len
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                img_size = seq_data['metadata']['shape']
                random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
                random_reverse = np.random.randint(0, 2) if self.randomize else 0
                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_z = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_z = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_y = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[2] - self.sub_seq_size[2] > 0:
                    crop_x = np.random.randint(0, img_size[2] - self.sub_seq_size[2]) if self.randomize else 0
                else:
                    crop_x = 0

                flip = np.random.randint(0, 2, 3) if self.randomize else [0, 0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if 'SIM' not in sequence_folder[0]:
                    edge_z = False
                else:
                    edge_z = True
                if self.augmentation:
                    affine_alpha = np.array([0.08] * 3)
                    elastic_alpha = np.array(self.sub_seq_size) * 2
                    elastic_sigma = np.array(self.sub_seq_size) * 0.15

                    if not edge_z:
                        affine_alpha[0] = 0.
                        elastic_alpha[0] = 0.

                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size, affine_alpha)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size, elastic_alpha,
                                                                  elastic_sigma,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['metadata']['filelist'])))
                if random_reverse:
                    filename_idx.reverse()
                if random_sub_sample:
                    filename_idx = filename_idx[::random_sub_sample]
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_z_stop = crop_z + self.sub_seq_size[0]
                crop_y_stop = crop_y + self.sub_seq_size[1]
                crop_x_stop = crop_x + self.sub_seq_size[2]

                processed_image = []
                processed_seg = []
                full_seg_seq = []
                processed_dist = []
                is_last = []
                img = seg = None
                # start_time = time.time()
                for t, file_idx in enumerate(filename_idx):
                    all_times = [time.time()]
                    if img is not None:
                        del img
                        del seg
                    img, seg, full_seg = self.read_images(sequence_folder, file_idx, start_z=crop_z,
                                                          stop_z=crop_z_stop)
                    img_crop = img[:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    seg_crop = seg[:, crop_y:crop_y_stop, crop_x:crop_x_stop]

                    if self.augmentation:
                        trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                        img_crop = trans_img
                        if np.any(np.isnan(img_crop)):
                            raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if np.any(np.isinf(img_crop)):
                            raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if not np.equal(seg_crop, -1).all():

                            # seg_not_valid = np.equal(seg_crop, -1)

                            trans_seg = self._get_transformed_image_(seg_crop.astype(np.float32), affine_matrix,
                                                                     indices, seg=True)
                            # trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                            #                                                affine_matrix,
                            #                                                indices, seg=True)
                            trans_seg_fix = self._fix_transformed_segmentation(trans_seg, edge_z)
                            # trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                            seg_crop = trans_seg_fix
                            # seg_crop[trans_not_valid] = -1
                            if np.any(np.isnan(seg_crop)):
                                raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                            if np.any(np.isinf(seg_crop)):
                                raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                    else:
                        seg_crop = self._fix_transformed_segmentation(seg_crop, edge_z)
                    if flip[0]:
                        img_crop = np.flip(img_crop, 0)
                        seg_crop = np.flip(seg_crop, 0)
                    if flip[1]:
                        img_crop = np.flip(img_crop, 1)
                        seg_crop = np.flip(seg_crop, 1)
                    if flip[2]:
                        img_crop = np.flip(img_crop, 2)
                        seg_crop = np.flip(seg_crop, 2)
                    if rotate > 0:
                        img_crop = np.rot90(img_crop, rotate, axes=(1, 2))
                        seg_crop = np.rot90(seg_crop, rotate, axes=(1, 2))
                    if self.return_dist:
                        if full_seg == -1:
                            dist_crop = np.zeros(self.dist_sub_seq_size)
                        else:
                            dist_crop, _ = self._gt2dist_(seg_crop)
                        all_times.append(time.time())
                        processed_dist.append(dist_crop)
                    is_last_frame = 1. if (t + 1) < len(filename_idx) else 0.
                    is_last.append(is_last_frame)
                    processed_image.append(img_crop)
                    processed_seg.append(seg_crop)
                    full_seg_seq.append(max(0, full_seg))
                    if self.coord.should_stop():
                        return

                try:
                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(0.5)

                    # print("Thread Time: {}".format(time.time() - start_time))

                    q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            print(str(err))
            self.coord.request_stop()
            print('This error:')
            raise err

    def _create_queues(self):

        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.device('/cpu:0'):
            with tf.name_scope('DataHandler'):
                dtypes = [tf.float32] * 4
                shapes = [self.sub_seq_size] * 2 + [()] * 2
                q_list = []
                q_stat_list = []
                for b in range(self.batch_size):
                    q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes,
                                           name='data_q_{}'.format(b))
                    q_list.append(q)
                    q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    @tf.function
    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            dist_list = []
            full_seg_list = []
            is_last_list = []
            for q in self.q_list:
                if self.return_dist:
                    img, seg, dist, full_seg, is_last = q.dequeue_many(self.unroll_len)
                    dist_list.append(dist)
                else:
                    img, seg, full_seg, is_last = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            dist_batch = tf.stack(dist_list, axis=0) if self.return_dist else None

            if self.data_format == 'NDHWC':
                image_batch = tf.expand_dims(image_batch, 5)
                seg_batch = tf.expand_dims(seg_batch, 5)
            elif self.data_format == 'NCDHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
            else:
                raise ValueError()

            # image_batch_list = tf.unstack(image_batch, num=self.unroll_len, axis=0)
            # seg_batch_list = tf.unstack(seg_batch, num=self.unroll_len, axis=0)
            # full_seg_batch_list = tf.unstack(full_seg_batch, num=self.unroll_len, axis=0)

        if self.return_dist:
            # dist_batch_list = tf.unstack(dist_batch, num=self.unroll_len, axis=0)
            return image_batch, seg_batch, full_seg_batch, is_last_batch, dist_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        self.coord = coord
        threads = []
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        threads.append(t)
        self.coord.register_thread(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


class CTCInferenceReader(object):

    def __init__(self, data_path, filename_format='t*.tif', normalize=True, pre_sequence_frames=0):

        file_list = glob.glob(os.path.join(data_path, filename_format))
        if len(file_list) == 0:
            raise ValueError('Could not read images from: {}'.format(os.path.join(data_path, filename_format)))

        def gen():
            file_list.sort()

            file_list_pre = file_list[:pre_sequence_frames].copy()
            file_list_pre.reverse()
            full_file_list = file_list_pre + file_list
            for file in full_file_list:
                img = cv2.imread(file, -1).astype(np.float32)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(file))
                if normalize:
                    img = (img - img.mean())
                    img = img / (img.std())
                yield img

        self.dataset = tf.data.Dataset.from_generator(gen, tf.float32)

    @classmethod
    def unit_test(cls):
        data_path = '/Users/aarbelle/Documents/CellTrackingChallenge/Training/DIC-C2DH-HeLa/01'
        filename_format = 't*.tif'
        normalize = True
        data_cls = cls(data_path, filename_format, normalize)
        for img, fname in data_cls.dataset:
            print(fname, img.shape, img.numpy().max(), img.numpy().min(), img.numpy().mean(), img.numpy().std())



if __name__ == "__main__":
    CTCSegReaderSequence3D.unit_test()
    # LSCRAMReader2D.unit_test()
    # CTCInferenceReader.unit_test()

