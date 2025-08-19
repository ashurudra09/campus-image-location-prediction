# Campus Image Location Prediction

This project uses a deep learning model to predict the geographical location (latitude and longitude) and region ID of images taken on a campus. It leverages a modified `ConvNext-Tiny` architecture to perform these distinct prediction tasks.

## Features

*   **Latitude & Longitude Prediction**: Regresses the precise GPS coordinates of an image.
*   **Region ID Classification**: Classifies the image into one of 15 predefined campus regions.

## Getting Started

Follow these steps to set up the project environment and run the models.

### Prerequisites

*   Python 3.8+
*   Conda or a virtual environment tool

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashurudra09/campus-image-location-prediction.git
    cd campus-image-location-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Data and Models:**
    *   Download the full dataset from [this link](https://drive.google.com/drive/folders/1vtLNqk0N2GriYKcv6FUwiPQHSCBN69E-?usp=sharing). Unzip and place the contents into the `data/raw/` directory.
    *   Download the pre-trained model weights from [this link](https://drive.google.com/drive/folders/171x8YmkyWHHWd6qQYW2Hr8iiDowahNCd?usp=sharing). Place the `.pth` files into the `models/` directory.


## Methodology

The core of this project is a `ConvNext-Tiny` model, pre-trained on ImageNet, which has been fine-tuned for 2 different tasks.

## About The Code:

### Preprocessing
For latlong task, IQR method is applied for preprocessing which removes the outliers by removing values below 1st quartile and above 3rd quartile. The latitude and longitude for training and validation are scaled using RobustScaling before being passed as inputs. 

For direction task, all angles outside of [0,360) were removed.

For regionID, labels are converted from 0-14 during training.

### Transforms used
Conversion to 224x224, followed by random rotation and normalization by ImageNet mean and std.

### Latlong Task
- Backbone used: ConvNext-Tiny with pretrained weights
- Modification: Classifier is modified by changing it to a layer norm, flatten, dropout of 0.5 and output to 2 nodes (latitude+longitude).
- MSE Loss with Adam optimizer and ReduceLROnPlateau schedulers are used.
- Hyperparameters used: Epochs = 50, LR = 1e-4 , Dropout =0.5 , Stochastic_prob_depth = 0.5 (To avoid overfitting)
- Early stopping when LR drops below 1e-6

### RegionID Task
- Backbone used: ConvNext-Tiny with pretrained weights
- Modification: Classifier is modified by changing it to a layer norm, flatten, dropout of 0.5 and output to 15 node (one for each region ID).
- CrossEntropy Loss with Adam optimizer and ReduceLROnPlateau schedulers are used.
- Hyperparameters used: Epochs = 50, LR = 1e-4 , Dropout =0.5 , Stochastic_prob_depth = 0.5 (To avoid overfitting)
- Early stopping when LR drops below 1e-6

### Direction Task
- Backbone used: ConvNext-Tiny with pretrained weights
- Modification: Classifier is modified by changing it to a layer norm, flatten, dropout of 0.5 and output to 1 node.
- MAAE Loss with Adam optimizer and ReduceLROnPlateau schedulers are used.
- Hyperparameters used: Epochs = 50, LR = 1e-4 , Dropout =0.5 , Stochastic_prob_depth = 0.5 (To avoid overfitting)
- Early stopping when LR drops below 1e-6

---

### Link to Model Weights
[https://drive.google.com/drive/folders/1j_WcoJo53CN_7BNkfpWw2KUbiK9SO1pm?usp=sharing ](https://drive.google.com/drive/folders/171x8YmkyWHHWd6qQYW2Hr8iiDowahNCd?usp=sharing)


### Link to Dataset
[https://drive.google.com/drive/folders/1vtLNqk0N2GriYKcv6FUwiPQHSCBN69E-?usp=sharing](https://drive.google.com/drive/folders/1vtLNqk0N2GriYKcv6FUwiPQHSCBN69E-?usp=sharing)

---