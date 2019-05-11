# LSTM-UNet

The code in this repository is suplamentary to our paper "Microscopy Cell Segmentation via Convolutional LSTM Networks" published in ISBI 2019.
If this code is used please cite the paper:

@article{arbelle2018microscopy,
  title={Microscopy Cell Segmentation via Convolutional LSTM Networks},
  author={Arbelle, Assaf and Raviv, Tammy Riklin},
  journal={arXiv preprint arXiv:1805.11247},
  year={2018}
}

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


This project is writen in Python 3 and makes use of tensorflow 2.0.0a0. 
Please see the requierments.txt file for all prerequisits. 

### Installing

In order to get the code, either clone the project, or download a zip from GitHub:
```
git clone https://github.com/arbellea/LSTM-UNet.git
```

Install all python requierments

```
pip3 install -r <PATH TO REPOSITORY>/requirements.txt 
```

This should do it!
### Data

The training script was tailored for the Cell Tracking Benchmarck
If you do not have the training data and wish to train on the challenge data please contact the organizers through the website: www.celltrackingchallenge.net
Once you have the data, untar the file metadata_file.tar.gz into the direcroty of the training data: 

```
cd <PATH TO CELLTRACKINGCHALLENGE DATA>/Training
tar -xzvf  <PATH TO REPOSITORY>/metadata_files.tar.gz
```
make sure that metadata_01.pickle and metadata_02.pickle are located in each dataset directory (Only of 2D datasets)
## Training
### Modify Parameters

Open the Params.py file and change the paths for ROOT_DATA_DIR and ROOT_SAVE_DIR. 
ROOT_DATA_DIR should point to the directory of the cell tracking challenge training data: <PATH TO CELLTRACKINGCHALLENGE DATA>/Training and ROOT_SAVE_DIR should point to whichever directory you would like to save the checkpoints and tensorboard logs.

  

### Run Training Script:
In order to set the parameters for training you could either change the parameters if Params.py file under class CTCParams
or input them through command line.
You are encourged to go over the parameters in CTCParams to see the avialable options
The training script is train2D.py

```
python3 train2D.py
```
### Training on Private Data:
Since there are many formats for data and many was to store annotations, we could note come up with a generic data reader.
So if one would like to train on private data we recommend one of the following:
1. Save the data in the format of the cell tracking challenge and create the corresponding metadata_<sequenceNumber>.pickle file. 
2. Write you own Data reader with similar api to ours. See the data reader in DataHandling.py


## Authors

Assaf Arbelle (arbellea@post.bgu.ac.il) and Tammy Riklin Raviv (rrtammy@ee.bgu.ac.il)
Ben Gurion University of the Negev
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
