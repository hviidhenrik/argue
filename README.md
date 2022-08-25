## ARGUE anomaly detection
ARGUE is the abbreviation of Anomaly detection by Recombining Gated Unsupervised Experts. This is a model for detecting
anomalous data samples based on a combination of neural networks working together based on an approach called 
Mixture of Experts. The model itself is based on the 2020 paper by Schulze and Sperl: https://arxiv.org/abs/2008.13763. Their original code implementing the model can be found at: https://github.com/Fraunhofer-AISEC/ARGUE

ARGUE is based on the idea of analysing the hidden activation patterns of several autoencoder networks, which are responsible for each their own partition of the data. 

The model is only trained on normal/healthy/nominal data points, so the activation patterns are learned for healthy data. 
When new data comes in the model analyses the activation patterns these yield. The hypothesis is that data that is sufficiently different from the 
training data will result in different activation patterns. The idea of using the activation patterns is to allow for the 
detection of much more subtle anomalies rather than the usual way, using the prediction errors of a single autoencoder, where 
subtle anomalous patterns may well have been suppressed during the encoding and decoding of the data. 

The repo organization (somewhat) follows the cookiecutter structure given at: 
https://drivendata.github.io/cookiecutter-data-science/#directory-structure



### Ideas
TO DO
 - make AUC evaluation
   - :heavy_check_mark: implemented

Nice to have
 - a clustering method could be standard partitioning method, if no class vector is given
   - :heavy_check_mark: implemented
 - make data handling more clean (maybe make a class for handling this)
 - more realistic anomalies for the noise counter examples - think VAE or AAE

Experimental
 - could the raw alarm probabilities be used without the gating if we simply take the minimum probability over all
   the models for each datapoint?
   - :heavy_check_mark: ARGUELite accomplishes this
 - could data be sliced vertically instead of horizontally? So each decoder is responsible for a
   predetermined set of variables instead of rows? Could also be used to model several pumps at the same time, or
   have several submodels inside one ARGUE model
 - look into variable importance / fault contribution analysis
 - look into decorrelating the variables in the latent space

Speedups
 - once autoencoder is trained, simply extract activations from it for each datapoint and train alarm&gating on these,
   instead of calling the activation models every iteration
   - :heavy_check_mark: ARGUELite accomplishes this

