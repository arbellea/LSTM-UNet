import argparse
import os
import pickle
# noinspection PyPackageRequirements
import tensorflow as tf
import Networks as Nets
import Params
import DataHandeling
import sys
import losses
from utils import log_print
import requests

__author__ = 'arbellea@post.bgu.ac.il'

try:
    # noinspection PyPackageRequirements
    import tensorflow.python.keras as k
except AttributeError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import tensorflow.keras as k

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(f'Using Tensorflow version {tf.__version__}')
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')


class AWSError(Exception):
    pass


def train():
   
    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        # Data input
        train_data_provider = params.train_data_provider
        val_data_provider = params.val_data_provider
        coord = tf.train.Coordinator()
        train_data_provider.start_queues(coord)
        val_data_provider.start_queues(coord)

        # Model

        model = params.net_model(params.net_kernel_params, params.data_format, False)

        # Losses and Metrics

        ce_loss = losses.WeightedCELoss(params.channel_axis + 1, params.class_weights)
        seg_measure = losses.seg_measure(params.channel_axis + 1, three_d=False)
        train_loss = k.metrics.Mean(name='train_loss')
        train_seg_measure = k.metrics.Mean(name='train_seg_measure')
        train_accuracy = k.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_loss = k.metrics.Mean(name='val_loss')
        val_accuracy = k.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        val_seg_measure = k.metrics.Mean(name='val_seg_measure')

        # Save Checkpoints
        optimizer = tf.compat.v2.keras.optimizers.Adam(lr=params.learning_rate)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, net=model)
        if params.load_checkpoint:

            if os.path.isdir(params.load_checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(params.load_checkpoint_path)
            else:
                latest_checkpoint = params.load_checkpoint_path
            try:
                print(latest_checkpoint)
                if latest_checkpoint is None or latest_checkpoint == '':
                    log_print("Initializing from scratch.")
                else:
                    ckpt.restore(latest_checkpoint)
                    log_print("Restored from {}".format(latest_checkpoint))

            except tf.errors.NotFoundError:
                raise ValueError("Could not load checkpoint: {}".format(latest_checkpoint))

        else:
            log_print("Initializing from scratch.")

        manager = tf.train.CheckpointManager(ckpt, os.path.join(params.experiment_save_dir, 'tf_ckpts'),
                                             max_to_keep=params.save_checkpoint_max_to_keep,
                                             keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)

        @tf.function
        def train_step(image, label):
            with tf.GradientTape() as tape:
                predictions, softmax = model(image, True)
                loss = ce_loss(label, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            ckpt.step.assign_add(1)
            train_loss(loss)
            seg_value = seg_measure(label, predictions)
            if params.channel_axis == 1:
                predictions = tf.transpose(predictions, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            train_accuracy(label, predictions)
            train_seg_measure(seg_value)
            return softmax, predictions, loss

        @tf.function
        def val_step(image, label):
            predictions, softmax = model(image, False)
            t_loss = ce_loss(label, predictions)

            val_loss(t_loss)
            seg_value = seg_measure(label, predictions)
            if params.channel_axis == 1:
                predictions = tf.transpose(predictions, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            val_accuracy(label, predictions)
            val_seg_measure(seg_value)
            return softmax, predictions, t_loss

        train_summary_writer = val_summary_writer = train_scalars_dict = val_scalars_dict = None
        if not params.dry_run:
            train_log_dir = os.path.join(params.experiment_log_dir, 'train')
            val_log_dir = os.path.join(params.experiment_log_dir, 'val')
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            train_scalars_dict = {'Loss': train_loss, 'SEG': train_seg_measure}
            val_scalars_dict = {'Loss': val_loss, 'SEG': val_seg_measure}

        def tboard(writer, step, scalar_loss_dict, images_dict):
            with tf.device('/cpu:0'):
                with writer.as_default():
                    for scalar_loss_name, scalar_loss in scalar_loss_dict.items():
                        tf.summary.scalar(scalar_loss_name, scalar_loss.result(), step=step)
                    for image_name, image in images_dict.items():
                        if params.channel_axis == 1:
                            image = tf.transpose(image, (0, 2, 3, 1))
                        tf.summary.image(image_name, image, max_outputs=1, step=step)

        template = '{}: Step {}, Loss: {}, Accuracy: {}'
        try:
            # if True:
            val_states = model.get_states()
            train_imgs_dict = {}
            val_imgs_dict = {}
            for _ in range(int(ckpt.step), params.num_iterations + 1):
                if params.aws:
                    r = requests.get('http://169.254.169.254/latest/meta-data/spot/instance-action')
                    if not r.status_code == 404:
                        raise AWSError('Quitting Spot Instance Gracefully')

                image_sequence, seg_sequence, _, is_last_batch = train_data_provider.get_batch()
                if params.profile:
                    tf.summary.trace_on(graph=True, profiler=True)
                train_output_sequence, train_predictions, train_loss_value = train_step(image_sequence, seg_sequence)
                # q_stats = [qs().numpy() for qs in params.train_data_provider.q_stat_list]
                # print(q_stats)
                if params.profile:
                    with train_summary_writer.as_default():
                        tf.summary.trace_export('train_step', step=int(ckpt.step),
                                                profiler_outdir=params.experiment_log_dir)
                model.reset_states_per_batch(is_last_batch)  # reset states for sequences that ended

                if not int(ckpt.step) % params.write_to_tb_interval:
                    if not params.dry_run:

                        seg_onehot = tf.one_hot(tf.cast(tf.squeeze(seg_sequence[:, -1], params.channel_axis), tf.int32),
                                                depth=3)
                        if params.channel_axis == 1:
                            seg_onehot = tf.transpose(seg_onehot, (0, 3, 1, 2))
                        display_image = image_sequence[:, -1]
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        train_imgs_dict['Image'] = display_image
                        train_imgs_dict['GT'] = seg_onehot
                        train_imgs_dict['Output'] = train_output_sequence[:, -1]
                        tboard(train_summary_writer, int(ckpt.step), train_scalars_dict, train_imgs_dict)
                        log_print('Printed Training Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")

                if int(ckpt.step) % params.save_checkpoint_iteration == 0 or int(ckpt.step) == params.num_iterations:
                    if not params.dry_run:
                        save_path = manager.save(int(ckpt.step))
                        log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    else:
                        log_print("WARNING: dry_run flag is ON! Mot saving checkpoints or tensorboard data")
                if not int(ckpt.step) % params.print_to_console_interval:
                    log_print(template.format('Training', int(ckpt.step),
                                              train_loss.result(),
                                              train_accuracy.result() * 100))

                if not int(ckpt.step) % params.validation_interval:
                    train_states = model.get_states()
                    model.set_states(val_states)
                    (val_image_sequence, val_seg_sequence, _, val_is_last_batch,
                     ) = val_data_provider.get_batch()
                    val_output_sequence, val_predictions, val_loss_value = val_step(val_image_sequence,
                                                                                    val_seg_sequence)
                    model.reset_states_per_batch(val_is_last_batch)  # reset states for sequences that ended
                    if not params.dry_run:
                        seg_onehot = tf.one_hot(tf.cast(tf.squeeze(val_seg_sequence[:, -1], params.channel_axis),
                                                        tf.int32), depth=3)
                        if params.channel_axis == 1:
                            seg_onehot = tf.transpose(seg_onehot, (0, 3, 1, 2))
                        display_image = val_image_sequence[:, -1]
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        val_imgs_dict['Image'] = display_image
                        val_imgs_dict['GT'] = seg_onehot
                        val_imgs_dict['Output'] = val_output_sequence[:, -1]
                        tboard(val_summary_writer, int(ckpt.step), val_scalars_dict, val_imgs_dict)
                        log_print('Printed Validation Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")

                    log_print(template.format('Validation', int(ckpt.step),
                                              val_loss.result(),
                                              val_accuracy.result() * 100))
                    val_states = model.get_states()
                    model.set_states(train_states)

        except (KeyboardInterrupt, ValueError, AWSError) as err:
            if not params.dry_run:
                log_print('Saving Model Before closing due to error: {}'.format(str(err)))
                save_path = manager.save(int(ckpt.step))
                log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                # raise err

        except Exception as err:
            #
            raise err
        finally:
            if not params.dry_run:
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt'.format(int(ckpt.step)))
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (params.net_kernel_params,)},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                log_print('Saved Model to file: {}'.format(model_fname))
            else:
                log_print('WARNING: dry_run flag is ON! Not Saving Model')
            log_print('Closing gracefully')
            coord.request_stop()
            coord.join()
            log_print('Done')


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
                datastets.append((values[i], (values[i + 1])))
            setattr(namespace, self.dest, datastets)


    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('-n', '--experiment_name', dest='experiment_name', type=str,
                            help="Name of experiment")
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                            help="Visible GPUs: example, '0,2,3'")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    arg_parser.add_argument('--profile', dest='profile', type=bool,
                            help="Write profiling data to tensorboard. For debugging only")
    arg_parser.add_argument('--root_data_dir', dest='root_data_dir', type=str,
                            help="Root folder containing training data")
    arg_parser.add_argument('--data_provider_class', dest='data_provider_class', type=str, action=AddReader,
                            help="Type of data provider")
    arg_parser.add_argument('--dataset', dest='train_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs: DatasetName, SequenceNumber")
    arg_parser.add_argument('--val_dataset', dest='val_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs DatasetName, SequenceNumber")
    arg_parser.add_argument('--net_gpus', dest='net_gpus', type=int, nargs='+',
                            help="gpus for each net: example: 0 0 1")
    arg_parser.add_argument('--net_types', dest='net_types', type=int, nargs='+', action=AddNets,
                            help="Type of nets")
    arg_parser.add_argument('--crop_size', dest='crop_size', type=int, nargs=2,
                            help="crop size for y and x dimensions: example: 160 160")
    arg_parser.add_argument('--train_q_capacity', dest='train_q_capacity', type=int,
                            help="Capacity of training queue")
    arg_parser.add_argument('--val_q_capacity', dest='val_q_capacity', type=int,
                            help="Capacity of validation queue")
    arg_parser.add_argument('--num_train_threads', dest='num_train_threads', type=int,
                            help="Number of train data threads")
    arg_parser.add_argument('--num_val_threads', dest='num_val_threads', type=int,
                            help="Number of validation data threads")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")
    arg_parser.add_argument('--batch_size', dest='batch_size', type=int,
                            help="Batch size")
    arg_parser.add_argument('--unroll_len', dest='unroll_len', type=int,
                            help="LSTM unroll length")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--validation_interval', dest='validation_interval', type=int,
                            help="Number of iterations between validation iteration")
    arg_parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_const', const=True,
                            help="Load from checkpoint")
    arg_parser.add_argument('--load_checkpoint_path', dest='load_checkpoint_path', type=str,
                            help="path to checkpoint, used only with --load_checkpoint")
    arg_parser.add_argument('--continue_run', dest='continue_run', action='store_const', const=True,
                            help="Continue run in existing directory")
    arg_parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                            help="Learning rate")
    arg_parser.add_argument('--class_weights', dest='class_weights', type=float, nargs=3,
                            help="class weights for background, foreground and edge classes")
    arg_parser.add_argument('--save_checkpoint_dir', dest='save_checkpoint_dir', type=str,
                            help="root directory to save checkpoints")
    arg_parser.add_argument('--save_log_dir', dest='save_log_dir', type=str,
                            help="root directory to save tensorboard outputs")
    arg_parser.add_argument('--tb_sub_folder', dest='tb_sub_folder', type=str,
                            help="sub-folder to save outputs")
    arg_parser.add_argument('--save_checkpoint_iteration', dest='save_checkpoint_iteration', type=int,
                            help="number of iterations between save checkpoint")
    arg_parser.add_argument('--save_checkpoint_max_to_keep', dest='save_checkpoint_max_to_keep', type=int,
                            help="max recent checkpoints to keep")
    arg_parser.add_argument('--save_checkpoint_every_N_hours', dest='save_checkpoint_every_N_hours', type=int,
                            help="keep checkpoint every N hours")
    arg_parser.add_argument('--write_to_tb_interval', dest='write_to_tb_interval', type=int,
                            help="Interval between writes to tensorboard")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    print(args_dict)
    params = Params.CTCParams(args_dict)
    # params = Params.CTCParamsNoLSTM(args_dict)

    # try:
    #     train()
    # finally:
    #     log_print('Done')
    train()
