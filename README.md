# ssl-for-echocardiograms
Code for the MLHC 2021 paper: A New Semi-supervised Learning Benchmark for Classifying View and Diagnosing Aortic Stenosis from Echocardiograms

# Demo
[load pretrained checkpoint and inference](LoadCheckpoint_Demo.ipynb)


# Setup
### Download dataset
Please visit our website https://www.eecs.tufts.edu/~mhughes/research/tmed/ and download the data

### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Create environment
conda env create -f environment.yml

### Process the dataset
PROVIDE A DEMO NOTEBOOK


# Running experiments
The commands for reproducing major tables in the paper are provided in [runs](runs/) 

### Image-level predictions

For example, if you want to run a fully supervised baseline on our suggested split1 on the TMED-18-18

```
bash launch_fs_fold0.sh run_here
```

### Patient-level predictions
We have provided the saved image level predictions in folder [predictions](predictions/), so that the patient level prediction code can run smoothly to reproduce our results in the paper. 

For example, if you want to run the patient level prediction for MixMatch on our suggested split1 on TMED-156-52, go to [runs/table7](runs/table7)

```
bash launch_MixMatch_fold0.sh run_here
```

In practice, the user can save their own image level predictions to the folder and run the script instead.

# Analysing results

The results including the training curves, validation and test balanced accuracy with and without ensemble, will be automatically saved under the results_analysis folder (created during training) under your specified train_dir in the bash script.  

### A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of TensorFlow used, random seeds, etc. 


# Citing this work
