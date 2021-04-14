# Feature Selection and Classification

This is an academic project. This repository walks through 5 different correlation metrics, and uses them for feature selection.
The repository provides option to combine metrics together if someone needs it. For classification,
SVM classifier from scikit-learn is used. If one wishes to search for best combination of
correlation metrics to use or even wishes to go for hyperparameter optimization for the classifier,
there's support for GridSearchCV or RandomizedSearchCV from scikit-learn which one can use.  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to install these softwares and libraries before you start

```bash
python3
sklearn==0.20.0
```

### Installing

[Python-3](https://www.python.org/downloads/) can be installed by downloading the package for free from the website: -

The sklearn python library can be installed using 
```bash
sudo apt-get install sklearn
```
If you have pip installed, you can also use it to install from the wheel.
```bash
pip install sklearn==0.20.0
```
or by installing from requirements.txt
```bash
pip install -r requirements.txt
```

## Running the code
The code allows you to run it in two modes: -
* Running the GridSearchCV or RandomizedSearchCV.
  
  If you wish to check the combination of correlation metrics that gives best results for a given value
  of dimensions to reduce to or the best parameter value to set the classifier to, you can run
  *gridsearch.py* file as below: -
  ```bash
  python gridsearch.py TRAIN_DATA_FILE_PATH TEST_DATA_FILE_PATH TRAINLABEL_FILE_PATH [OPTIONS]
  ```
  you can run help to see various options to run the file.
  ```bash
  python gridsearch.py --help
  ```
* Running the main file.
  
  If you wish to run one single run with a set of combination, and parameter settings for classifier,
  you can execute *main.py* file as below: -
  ```bash
  python main.py TRAIN_DATA_FILE_PATH TEST_DATA_FILE_PATH TRAINLABEL_FILE_PATH [OPTIONS]
  ```
  you can run help to see various options to run the file.
  ```bash
  python main.py --help
  ```

## Running the tests

The project currently lacks unittests and integration tests.

## Deployment

The project currently lacks a deployment script.

## Acknowledgments

* *Golub et. al.* for SNR correlation metric.  
* *Cho et. al.* for Mutual Information index.
* [*Billie Thompson et. al.*](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) for README template.
