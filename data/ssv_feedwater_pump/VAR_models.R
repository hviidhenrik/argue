rm(list=list())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# model <- define.model(kvar=5, ar=c(1), ma=0, indep=NULL)
# marima_model <- marima(df, means=0, ar.pattern = model$ar.pattern, 
#                        ma.pattern = model$ma.pattern, Check = F, Plot = "none",
#                        penalty = 0)

library(tidyverse)
library(marima)


# residual covariance matrix
sigma = matrix(c(1, 0.9, 0.8, 0, 0,
                 0.9, 1, 0.7, 0, 0,
                 0.8, 0.7, 1, 0, 0,
                 0, 0, 0, 1, 0.9,
                 0, 0, 0, 0.9, 1),
               nrow=5, byrow=T)

# the vector AR(1) coefficient matrix with leading unity matrix
ar_matrix = array(c(diag(5), matrix(c(0.88, rep(0,4),
                                      0, 0.58, rep(0,3),
                                      0, 0, 0.89, 0, 0,
                                      0, 0, 0, 0.66, 0,
                                      0, 0, 0, 0, 0.3),
                                    nrow=5, byrow=T)), dim=c(5,5,2))

# define helper function to generate VAR(1) timeseries data
make_var1_process <- function(n=100, ar_coef_matrix=diag(2), 
                              resid_cov_matrix=diag(dim(ar_coef_matrix)[2]), 
                              averages=rep(0,dim(ar_coef_matrix)[2])){
  library(marima)
  
  dim_variables <- dim(ar_coef_matrix)[2]
  
  y_simulated <- marima.sim(kvar=dim_variables, 
                            ar.model = ar_matrix,
                            ma.model = NULL,
                            averages = averages,
                            resid.cov = resid_cov_matrix,
                            nsim = n)
  return(y_simulated)
}

# make some data and plot it
df <- as_tibble(make_var1_process(500, ar_coef_matrix = ar_matrix, resid_cov_matrix = sigma))
summary(df)
cor(df)

ts <- as.ts(df)
plot(ts)
