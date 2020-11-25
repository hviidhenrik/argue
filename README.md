# Overview
This repository is a starter-kit for developing new models for the SmartPlant-PdM Team. 
It provides a project structure and some boilerplate code such as the sphinx documentation skeleton 
to get you started on your model development.

## Directory structure

```bash
├── README.md   <- Root README 
├── archive     <- Trained and serialized models
├── azure-pipelines.yml <- yaml file detailing azure pipeline
├── docs        <- Sphinx docs
├── make.bat    <- Make.bat sphinx
├── Makefile    <- MakeFile - mostly sphinx 
├── reports     <- Generated reports/figures
├── requirements.txt    <- the requirement file
├── setup.py    <- Distutils setup file
├── src         <- Source code
│   ├── data            <- Loading the data  
│   ├── data_exploration    <- code for initial data exploration 
│   ├── deployment    <- Home of deployment scripts adn configs
│   └── models              <- Scripts to train models and perform predictions 
├── tasks       <- Invoke tasks
├── templates   <- template holding e.g. twine upload config
├── tests       <- pytest
```

Most recent SmartPlant PdM wheel that provides base classes for model development can be found [here]((https://dongenergy-p.visualstudio.com/DefaultCollection/Bioenergy/_packaging?_a=package&feed=BioenergyFeed&package=smart-plant-predictive-maintenance&protocolType=PyPI&version=1.9.3))

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
