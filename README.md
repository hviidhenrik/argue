# SmartPlant Predictive Maintenance PhD repo
This repository is the home of the industrial PhD project being carried out in the SmartPlant Predictive Maintenance (PdM) team in Bioenergy. 
The project aims to improve what we already do and find new and better ways of doing PdM together with the DTU Compute at the section for 
statistics and data analysis.

## ARGUE anomaly detection
ARGUE is the abbreviation of Anomaly detection by Recombining Gated Unsupervised Experts. This is a model for detecting
anomalous data samples based on a combination of neural networks working together based on an approach called 
Mixture of Experts. The model itself is based on the 2020 paper by Sperl and Schulze. It analyses the hidden activation
patterns of several autoencoder networks, which are responsible for each their own partition of the data. 

The model is only trained on normal/healthy/nominal data points, so the activation patterns are learned for healthy data. 
When new data comes in 
the model analyses the activation patterns these yield. The hypothesis is that data that is sufficiently different from the 
training data will result in different activation patterns. The idea of using the activation patterns is to allow for the 
detection of much more subtle anomalies rather than the usual way, using the prediction errors of a single autoencoder, where 
subtle anomalous patterns may well have been suppressed during the encoding and decoding of the data. 


The repo organization follows the cookiecutter structure given at: https://drivendata.github.io/cookiecutter-data-science/#directory-structure




