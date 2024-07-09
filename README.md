# HandSeals Recognition Tool

## Description

The HandSeals Recognition Tool is a deep learning-based application designed to recognize American Sign Language using multiple models including AlexNet, ResNet, ~~InceptionV1~~, and DenseNet.  
This tool provides a platform for training, evaluating, and making real-time predictions of hand seals, with a user-friendly graphical interface built using PyQt5.

## Installation Guide

To get started, follow these steps to set up the environment and install the necessary dependencies.

### Prerequisites
- Download and install Anaconda: [Anaconda Download](https://www.anaconda.com/download)

### Step-by-Step Instructions

1. #### **Create a new conda environment**

    ```bash
    conda create --name handseals python=3.9
    ```

2. #### **Activate the environment**

    ```bash
    conda activate handseals
    ```

    Or open **Anaconda Prompt (handseals)**.

3. #### **Install PyTorch and CUDA (if using Windows with an Nvidia GPU)**

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

    If you are not using an Nvidia GPU, proceed to the [PyTorch website](https://pytorch.org/) to install the appropriate version for your PC.

4. #### **Install additional packages**

    ```bash
    pip install opencv-python==4.7.0.72
    pip install PyQt5==5.15.10
    pip install numpy==1.24.4
    ```

5. #### **Install Visual Studio Code**

    Download and install Visual Studio Code: [Visual Studio Code](https://code.visualstudio.com/)

6. **Clone the repository**

    ```bash
    git clone https://github.com/H-qitai/HandSeals
    ```

7. #### **Navigate to the cloned repository**

    If the repository is in the Downloads folder:
    ```bash
    cd Downloads/HandSeals
    ```
    Otherwise, use `cd` to navigate to the appropriate directory.

8. #### **Obtain dataset**

    Unzip `dataset.zip`. You should obtain an Excel file that looks like the following:

    ![ExcelPic](https://github.com/H-qitai/HandSeals/blob/main/resources/excelPic.png)

    The labels correspond to different hand signs. Below is an image for a clearer view:

    ![Translated](https://github.com/H-qitai/HandSeals/blob/main/resources/translated.png)

9. #### **Run the application**

    ```bash
    python HandSeals.py
    ```

## Usage Guide

1. #### **Loading data**

    Click "Load Data" to load the Excel sheet that was unzipped earlier.

    ![View](https://github.com/H-qitai/HandSeals/blob/main/resources/view.png)

    The data is now loaded.

2. #### **Training the models**

    Click "Train" to select different parameters to train different AI models.

    ![Train](https://github.com/H-qitai/HandSeals/blob/main/resources/train.png)

    Click "Start Training" to start training.

    A graph showing accuracy and training loss will be displayed.

    After training is completed, the model will be saved.

3. #### **Loading trained models**

    Click "Test" to load the trained models.

    A page similar to "View" will show up.

    Click on the images to show the trained prediction result.

    An example is shown below:

    ![Prediction](https://github.com/H-qitai/HandSeals/blob/main/resources/prediction.png)

4. #### **Live Prediction**

    After loading a model, click "Open Camera" to use the live prediction feature.

    Press "c" to capture or "q" to quit.

    Make sure the hand sign is in the middle of the camera.

    ![Live](https://github.com/H-qitai/HandSeals/blob/main/resources/live.png)
