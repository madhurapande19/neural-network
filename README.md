# neural-network

This is a simple implementation of a Multilayer Feedforward Neural Network. This network will be trained and
tested using the A-Z Handwritten Alphabets dataset. Specifically, given an input image
(28 x 28 = 784 pixels) from the dataset, the network will be trained to classify the image
into 1 of 10 classes.(Randomly sampled). 

## Files
> 1.src/train.csv : The basic model which gets trained on the above task.
> 2.src/run.sh : Wrapper around train.csv, has all the hyperparameters tuned to the best value.
> 3.src/testing.sh : This will use the pretrained model and saves predictions on test data in expt_dir.
> 4.save_dir : This contains the weights of best model trained so far.

## Steps to train/test
> 1. Download train.csv from "https://drive.google.com/file/d/1ZviWz8h5Cw85d3lqIi1bOzH-nQh-HOJD/view?usp=sharing". All the data files," train.csv", "valid.csv" and "test.csv" should be in same folder as train.py.
> 2. To generated predictions using pretrained model, just run:
    bash testing.sh
> 3. To train the model from scratch, run:
    bash run.sh

