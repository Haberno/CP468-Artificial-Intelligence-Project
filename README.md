# CP468: Notebook Documentation:

## Project Overview

In this project we aim to build 3 pre trained models to detect Melanoma Skin Cancer. We trained our models by using 25331 images. In particular we used VGG16, ResNet50 and DenseNet201 with imagent as our weight for the data.

&nbsp;
# Documentation of How to Install and Prepare Dataset for Training

## Step 1: Data Acquisition

- #### The 25331 images used for the model were sourced from the [ISIC Challenge Dataset](https://challenge.isic-archive.com/data/#2019). The dataset is an amalgamation of 3 datasets; HAM10000, BCN_20000, and MSK Dataset.

## Step 2:

- #### Open the Training_Input_GroundTruth.csv and remove all columns except "MEL" and save the csv

## Step 3: Upload to Google Drive
- #### After modifying your csv upload both the zip and csv to your Google Drive

## Step 4: Install required libraries and imports

- #### Install gradio and import required libraries such as zip, drive, os, numpy, and etc.

```python
!pip install gradio
from google.colab import drive
```

- #### Mount your google drive

## Step 5: 

- #### Unzip and extract all data from training input dataset.

- #### Seggregate the images based whether the lesion is benign or malignant PS. If the value under MEL is 0 lesion is benign else it is malignant.

## Step 6:

- #### Use the image_dataset_from_directory function to split the dataset into training and validation datasets. In this case we did a 66.6% to 33.3% split.

- #### Set the resize the images to the required input shape, and also set batch size. 

- #### Perform preprocessing by normalizing the dataset and data augmentation.

&nbsp;

*Now the data is ready to be used in training all 3 pre-trained models and the model built from scratch.*

&nbsp;
# Documentation of How to Download, Configure and Train 3 Pre-Trained Models

## Step 1:
- To Download the models using the Keras API use the following imports:

```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet201
```

- #### These imports are present in **Step 4** of in '*Documentation of How to Install and Prepare Dataset for Training*'

## Step 2:

- Normalize the data and apply data augmentation. This time we are not directory adding the data augmentation layers to the model.

## Step 2:

- #### Define these models and load in the model with their optimized weights
```python
#Example Definition of one of the models
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```


- #### Freeze some trainable layers of the pretrained models
```python
for layer in vgg.layers[:10]:
    layer.trainable = False

for layer in resnet.layers[:25]:
    layer.trainable = False

for layer in densenet.layers[:200]:
    layer.trainable = False
```

## Step 3:
- Declare EarlyStopping callback to use while training the model to prevent overfitting.

- Compile the models using the adam optimizer and use the *sparse_categorical_crossentropy* loss function. We need to use this because the labels are provivded as integers. 

## Step 4:
- Define checkpoint callbacks for the models to save each model at highest validation accuracy before each model training.
- Train  the models for 10 epochs and save the model histories.
- Use matplotlib to plot the loss, accuracy, validation loss, and validation accuracy.

## Step 5:
- Test the model on test dataset using the evaluate function and finally plot confusion matrix.


## Conclusion

 *Hooray you made it to the end! I hope you have a wonderful day.*














