# project-python-group-22

## HandSeals Recognition Tool

## Description
The HandSeals Recognition Tool is a deep learning-based application designed to recognize hand seals using multiple models including AlexNet, ResNet, ~~InceptionV1~~, and DenseNet.  
This tool provides a comprehensive platform for training, evaluating, and making real-time predictions of hand seals, with a user-friendly graphical interface built using PyQt5.


## install guide
To get started, you will have to first download Anaconda. (https://www.anaconda.com/download)

Run the command prompt in Anaconda.

Type this in: new conda environment conda create --name cs302 python=3.9

To activate your environment from now on, you will need to type <br>
"conda activate cs302" <br>
Or open anaconda prompt (cs302) <br>
That is after you have set it up <br>

Now type this in if you are using windows and have a GPU: <br>
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia<br>
else go to <br>
https://pytorch.org/

Now type these command into the setup environment: <br>
pip install opencv-python==4.7.0.72 <br>
pip install PyQt5==5.15.10 <br>
pip install numpy==1.24.4 <br>

Install VS code

https://code.visualstudio.com/

After everything has been setup

cd project-python-group-22

python HandSeals.py

The program should popup



## Use guide
Heres a youtube video guide. <br>

https://www.youtube.com/watch?v=ZJEg09ZemNg


Make sure to load the CSV datasheet before training, or testing the results, or live translate (camera) feature.