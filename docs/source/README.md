# Overview
This repository is a starter-kit for developing new models for the SmartPlant-PdM Team. 
It provides a project structure and some boilerplate code such as the sphinx documentation skeleton 
to get you started on your model development.

## Directory structure

```bash
├── Makefile    <- MakeFile - mostly sphinx 
├── README.md   <- Root README 
├── archive     <- Trained and serialized models 
├── azure-pipelines.yml
├── docs        <- Sphinx docs
├── make.bat    <- Make.bat sphinx
├── reports     <- Generated reports/figures
├── requirements.txt    <- the requirement file
├── setup.py    <- Make.bat sphinx
├── src         <- Source code
│   ├── data            <- Loading the data  
│   ├── data_exploration    <- code for initial data exploration 
│   ├── model_deployment    <- Home of deployment scripts
│   └── models              <- Scripts to train models and perform predictions 
├── tasks       <- home of invoke tasks
├── tests       <- pytest
```

## HowTo

### Get started
Clone the repo. Delete the .git folder and start a new git project with ```git init```.
Install the local requirements

```bash
conda install --file requirements.txt
```
or
```bash
pip install -r requirements.txt
```

Show invoke tasks

```bash
invoke --list
```

## Build and Test

### Build documentation

You can build and inspect the documentation using the command

```bash
invoke sphinx.build
```
or 

```bash
invoke sphinx.run
```

### Run pytest suite

```bash
invoke test.pytest
```
## Code quality

See SmartPlant PDM team's [coding guidelines](https://dev.azure.com/dongenergy-p/Bioenergy/_wiki/wikis/Bioenergy.wiki?wikiVersion=GBwikiMaster&pagePath=%2FSmartPlant%2FPredictive%20Maintenance%2FCoding%20guidelines&pageId=3585)
