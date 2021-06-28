# ssl-for-echocardiograms
Code for the MLHC 2021 paper: A New Semi-supervised Learning Benchmark for Classifying View and Diagnosing Aortic Stenosis from Echocardiograms

## Demo
[load pretrained checkpoint and inference](LoadCheckpoint_Demo.ipynb)

## Setup
### Download dataset
Please visit our website https://www.eecs.tufts.edu/~mhughes/research/tmed/ and download the data

### Install dependencies
conda env create -f environment.yml

## Running experiments
### image level predictions

### patient level predictions

### A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of TensorFlow used, random seeds, etc. 
