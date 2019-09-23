import DataHandeling
import os
from datetime import datetime
import Networks as Nets

__author__ = 'arbellea@post.bgu.ac.il'


ROOT_DATA_DIR = '~/CellTrackingChallenge/Training/'
ROOT_TEST_DATA_DIR = '~/CellTrackingChallenge/Test/'
ROOT_SAVE_DIR = '~/LSTM-UNet-Outputs/'


class ParamsBase(object):
    aws = False

    def _override_params_(self, params_dict: dict):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        this_dict = self.__class__.__dict__.keys()
        for key, val in params_dict.items():
            if key not in this_dict:
                print('Warning!: Parameter:{} not in defualt parameters'.format(key))
            setattr(self, key, val)

    pass


class CTCParams(ParamsBase):
    # --------General-------------
    experiment_name = 'MyRun_SIM'
    gpu_id = 0  # set -1 for CPU or GPU index for GPU.

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderSequence2D
    root_data_dir = ROOT_DATA_DIR
    train_sequence_list = [('Fluo-N2DH-SIM+', '01'), ('Fluo-N2DH-SIM+', '02')]  # [('Dataset Name', 'SequenceNumber'), ('Dataset Name', 'SequenceNumber'), ]
    val_sequence_list = [('Fluo-N2DH-SIM+', '01'), ('Fluo-N2DH-SIM+', '02')]
    crop_size = (128, 128)  # (height, width) preferably height=width 
    batch_size = 5
    unroll_len = 4
    data_format = 'NCHW' # either 'NCHW' or 'NHWC'
    train_q_capacity = 200
    val_q_capacity = 200
    num_val_threads = 2
    num_train_threads = 8

    # -------- Network Architecture ----------
    net_model = Nets.ULSTMnet2D
    net_kernel_params = {
        'down_conv_kernels': [
            [(3, 128), (3, 128)],  # [(kernel_size, num_filters), (kernel_size, num_filters), ...] As many convolustoins in each layer
            [(3, 256), (3, 256)],
            [(3, 256), (3, 256)],
            [(3, 512), (3, 512)],
        ],
        'lstm_kernels': [
            [(5, 128)],  # [(kernel_size, num_filters), (kernel_size, num_filters), ...] As many C-LSTMs in each layer
            [(5, 256)],
            [(5, 256)],
            [(5, 512)],
        ],
        'up_conv_kernels': [
            [(3, 256), (3, 256)],   # [(kernel_size, num_filters), (kernel_size, num_filters), ...] As many convolustoins in each layer
            [(3, 128), (3, 128)],
            [(3, 64), (3, 64)],
            [(3, 32), (3, 32), (1, 3)],
        ],

    }

    # -------- Training ----------
    class_weights = [0.15, 0.25, 0.6] #[background, foreground, cell contour]
    learning_rate = 1e-5
    num_iterations = 1000000
    validation_interval = 1000
    print_to_console_interval = 10

    # ---------Save and Restore ----------
    load_checkpoint = False
    load_checkpoint_path = ''  # Used only if load_checkpoint is True
    continue_run = False
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 5000
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'LSTMUNet'
    write_to_tb_interval = 500
    save_log_dir = ROOT_SAVE_DIR

    # ---------Debugging-------------
    dry_run = False  # Default False! Used for testing, when True no checkpoints or tensorboard outputs will be saved
    profile = False

    def __init__(self, params_dict):
        self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.train_data_base_folders = [(os.path.join(ROOT_DATA_DIR, ds[0]), ds[1]) for ds in
                                        self.train_sequence_list]
        self.val_data_base_folders = [(os.path.join(ROOT_DATA_DIR, ds[0]), ds[1]) for ds in
                                      self.val_sequence_list]
        self.train_data_provider = self.data_provider_class(sequence_folder_list=self.train_data_base_folders,
                                                            image_crop_size=self.crop_size,
                                                            unroll_len=self.unroll_len,
                                                            deal_with_end=0,
                                                            batch_size=self.batch_size,
                                                            queue_capacity=self.train_q_capacity,
                                                            data_format=self.data_format,
                                                            randomize=True,
                                                            return_dist=False,
                                                            num_threads=self.num_train_threads
                                                            )
        self.val_data_provider = self.data_provider_class(sequence_folder_list=self.val_data_base_folders,
                                                          image_crop_size=self.crop_size,
                                                          unroll_len=self.unroll_len,
                                                          deal_with_end=0,
                                                          batch_size=self.batch_size,
                                                          queue_capacity=self.train_q_capacity,
                                                          data_format=self.data_format,
                                                          randomize=True,
                                                          return_dist=False,
                                                          num_threads=self.num_val_threads
                                                          )

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if self.load_checkpoint and self.continue_run:
            if os.path.isdir(self.load_checkpoint_path):
                if self.load_checkpoint_path.endswith('tf-ckpt') or self.load_checkpoint_path.endswith('tf-ckpt/'):
                    self.experiment_log_dir = self.experiment_save_dir = os.path.dirname(self.load_checkpoint_path)
                else:
                    self.experiment_log_dir = self.experiment_save_dir = self.load_checkpoint_path
            else:

                save_dir = os.path.dirname(os.path.dirname(self.load_checkpoint_path))
                self.experiment_log_dir = self.experiment_save_dir = save_dir
                self.load_checkpoint_path = os.path.join(self.load_checkpoint_path, 'tf-ckpt')
        else:
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.experiment_name,
                                                   now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.experiment_name,
                                                    now_string)
        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'


class CTCInferenceParams(ParamsBase):

    gpu_id = 0  # for CPU ise -1 otherwise gpu id
    model_path = './Models/LSTMUNet2D/PhC-C2DL-PSC/' # download from https://drive.google.com/file/d/1uQOdelJoXrffmW_1OCu417nHKtQcH3DJ/view?usp=sharing
    output_path = './tmp/output/PhC-C2DL-PSC/01'
    sequence_path = os.path.join(ROOT_TEST_DATA_DIR, 'PhC-C2DL-PSC/01/')
    filename_format = 't*.tif'  # default format for CTC

    data_reader = DataHandeling.CTCInferenceReader
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'

    FOV = 0 # Delete objects within boundary of size FOV from each side of the image
    min_cell_size = 10  # Delete objects with less than min_cell_size pixels
    max_cell_size = 100  # Delete objects with more than max_cell_size pixels
    edge_dist = 2  # Regard the nearest edge_dist pixels as foreground
    pre_sequence_frames = 4  # Initialize the sequence with first pre_sequence_frames played in reverse

    # ---------Debugging---------

    dry_run = False
    save_intermediate = True
    save_intermediate_path = output_path

    def __init__(self, params_dict: dict = None):
        if params_dict is not None:
            self._override_params_(params_dict)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path,'IntermediateImages', now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)




