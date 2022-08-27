## ARGUE anomaly detection
ARGUE is the abbreviation of Anomaly detection by Recombining Gated Unsupervised Experts. This is a model for detecting
anomalous data samples based on a combination of neural networks working together based on an approach called 
Mixture of Experts. The model itself is based on the 2020 paper by Schulze and Sperl: https://arxiv.org/abs/2008.13763. 
Their original code implementing the model can be found at: https://github.com/Fraunhofer-AISEC/ARGUE. 
The code in this repository is not affiliated with the authors of the original paper - this is my own attempt at 
implementing their proposed model.  

ARGUE is based on the idea of analysing the hidden activation patterns of several autoencoder networks, 
which are specialised on different disjoint parts of the data. 

The model is only trained on normal/healthy/nominal data points. Thus, the activation pattern distributions are learned 
for healthy data only. When new data is to be predicted, the model analyses the activation patterns resulting from 
these. If the new data is significantly different from the raining data, it  will result in different activation 
patterns. The idea of using the activation patterns is to allow for the detection of more subtle anomaly information.
The traditional way of detecting anomalies by autoencoders is to the reconstruction error of a given data point. The 
motivating hypothesis behind ARGUE is that subtle anomaly patterns may be visible in the patterns of the hidden 
activation values of both the encoder and decoder networks. This information may not be visible in the raw 
reconstruction errors as information is lost during encoding and subsequent decoding. 

This is the case since reconstruction error can be seen as an aggregation of the differences between the training 
data and new data distributions. As such, it does not convey nuanced information about the nature of the data point 
itself, only a binary outcome as to whether the predicted observation deviates from the training data or not. 
If an observation deviations for a reason - because it is an anomaly - this deviation may be subtle enough, 
that it might well drown out in the inherent noise of real-world data and as a consequence be overlooked. 

### Ideas
TO DO
 - make AUC evaluation
   - :heavy_check_mark: implemented - in fact, any metric can be given to the models now

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

