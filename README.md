# COVID-Efficientnet &rarr; a Pytorch upgrade of the COVID-Next and COVID-Net

Inspired by COVID-Next and the efficiency and mobility of Efficientnet, I are now open sourcing the upgraded Pytorch implementation of both called COVID-Efficientnet.

COVID-Efficientnet features an architecture that builds upon Efficientnet b7 architecture, an AutoML architecture for optimizing both accuracy and mobility.

## Requirements

To install all requirements, simply run `pip3 install -r requirements.txt`.
Code was tested with Python 3.7.

### Dataset

We have created a script that automates the dataset generation from the two sources referenced in the original repo. To generate the dataset, follow these steps:

1. Download the datasets listed below:
    * COVID ChestXray [dataset](https://github.com/ieee8023/covid-chestxray-dataset.git). Be aware this repository is constantly adding new images.
    * Kaggle RSNA pneumonia [dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
2. Run the `generate_dataset.py` script. Run `python3 generate_dataset.py -h` to see supported arguments.

The script will create a new folder with `train` and `test` subfolders where images are located, along with the two metadata files for both train and test subsets.



## Training

Training configuration is currently modified through the `config.py` module. Check it out before starting training.

`python3 train.py` command will run model training.


## Results

The following results were obtained on the dataset used in the original repo as of May 19 2020.
| Model                  | Accuracy |
|:-----------------:|:--------:|
| COVID-Net (Large) | 91.90%   |
| COVID-Next    | 94.76%   |
| **COVID-Efficientnet**    | **96.01%**   |

### Pretrained Weights

Pretrained weights are available at [weights](https://drive.google.com/open?id=1-uCQr7gcPUj2szKrDK1tSfv1BNj8XBHM)


### Minimal prediction example

You can find the minimal prediction example in `minimal_prediction.py`.
The example demonstrates how to load the model and use it to predict the disease type on the image.

