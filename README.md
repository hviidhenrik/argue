# SmartPlant Predictive Maintenance PhD repo
This repository is the home of the industrial PhD project being carried out in the SmartPlant Predictive Maintenance (PdM) team. 
The project aims to improve what we already do and find new and better ways of doing PdM together with the Technical University
of Denmark (DTU) at the Depart of Applied Mathematics and Computer Science, also known as DTU Compute. 

The repo houses several experimental projects with different purposes. Some of these are listed below. 

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


## AnoGen

AnoGen is an experimental framework aiming to generate synthetic anomalies for 
the purpose of evaluating the skill of anomaly detection algorithms if no real anomalous data exists.
The framework requires only data representing a notion of a normal condition, 
and as such, no labels or known anomalies are required.


In short, what it does, is to train a model (a Variational Autoencoder) to recognize the normal condition well 
and learn its multivariate probability distribution by inferring a lower dimensional 
latent variable representation, which can be thought of as the generating mechanism that
underlies the normal data. Having learned the generating mechanism, we can draw samples from
the latent space that are outside the boundaries of the normal condition. Subsequently, these
can be transformed back to the original high-dimensional feature space where they will then
represent data that is different from the normal condition. 


Synthetic anomalies in hand, these can be used for evaluating how sensitive a trained 
anomaly detection algorithm is by letting it classify these anomaly points, which should
all be flagged as anomalies. On the other hand, the data representing the normal condition 
should not be flagged as anomalous. Having data representing both normal and anomalous, it
is then possible to finetune the algorithm's anomaly threshold and optimize metrics such as 
the false positive rate (FPR), sensitivity, accurracy etc.


In summary, the following steps are taken


Require: Normal data _D_

1. Train VAE(x) on _D_ s.t. avg(|VAE(x) - x_reconstructed|) < _epsilon_ and obtain _Decoder_ 
needed to reconstruct latent samples.
2. Train clustering C(_z_) on learned latent representation z_normal
3. Draw uniform random variables U ~ Unif(z_min, z_max) covering the learned latent 
space and an extra distance around it reflected in z_min and z_max.
4. Filter 1: reconstruct the samples U, i.e. _Decoder(U)_ = x_hat and remove those x_reconstructed 
samples that are outside the limits defined by the training data (and possibly a domain filter) +-
some small percentage buffer to reflect that anomalies are not necessarily restricted to the domain
of the training data. This yields a subset of samples, x_hat_realistic, considered 
reasonable/realistic. Keep the latent space coordinates of these, call them z_realistic. 
5. Filter 2: Use clustering C on the latent values z_realistic to identify those points that
are within the clusters of the normal data z_normal and remove them from z_realistic and their
corresponding values in x_hat_realistic to create x_hat_final, further 
reducing the subset of samples we want to keep. This makes sure we don't train our AD algorithm 
to flag normal data as anomalous (false positives)


Return x_hat_final






