# AnoGen

This repo is the home of an experimental framework aiming to generate synthetic anomalies for 
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






