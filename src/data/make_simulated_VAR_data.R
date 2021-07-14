rm(list=ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(marima)

### in case simulated data based on realistic data is desired, uncomment this:
# model <- define.model(kvar=5, ar=c(1), ma=0, indep=NULL)
# marima_model <- marima(df, means=0, ar.pattern = model$ar.pattern, 
#                        ma.pattern = model$ma.pattern, Check = F, Plot = "none",
#                        penalty = 0)


# --------------------------------------------------------------------------
set.seed(1234)

# simulate data from a vector AR(1) model
# define residual covariance matrix
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
  dim_variables <- dim(ar_coef_matrix)[2]
  
  y_simulated   <- marima.sim(kvar=dim_variables, 
                              ar.model = ar_matrix,
                              ma.model = NULL,
                              averages = averages,
                              resid.cov = resid_cov_matrix,
                              nsim = n)
  return(y_simulated)
}

# make phase1 data and verify correlation structure
N_phase1  <- 5000
df_phase1 <- as_tibble(make_var1_process(N_phase1, 
                                               ar_coef_matrix = ar_matrix, 
                                               resid_cov_matrix = sigma))
df_phase1$sample <- seq(1:dim(df_phase1)[1])
df_phase1$faulty  = 0
df_phase1        <- df_phase1[,c(7,6,seq(1:5))]

summary(df_phase1)
cor(df_phase1[,3:7])
ts <- as.ts(df_phase1)
plot(ts)

# make and inspect phase2 data
N_phase2  <- 500
df_phase2 <- as_tibble(make_var1_process(N_phase2, 
                                         ar_coef_matrix = ar_matrix,
                                         resid_cov_matrix = sigma))
df_phase2$sample <- seq(1:dim(df_phase2)[1])
df_phase2$faulty = 0
df_phase2        <- df_phase2[,c(6,7,seq(1:5))]

summary(df_phase2)
cor(df_phase2[,3:7])
ts <- as.ts(df_phase2)
plot(ts)

# inject anomaly signal after 100 samples
change_point                        <- 100
sd_v1                               <- sd(df_phase2$V1)
df_phase2$V1[change_point:N_phase2] <- df_phase2$V1[change_point:N_phase2] + 2*sd_v1

idx_faulty                  <- which(df_phase2$sample >= change_point)
df_phase2$faulty[idx_faulty] = 1

ts <- as.ts(df_phase2)
plot(ts)

# save to csv
path <- "..\\..\\data\\simulated_data\\"
df_phase1 %>% write_csv(file = paste0(path, "data_simulated_VAR1_phase1.csv"))
df_phase2 %>% write_csv(file = paste0(path, "data_simulated_VAR1_phase2.csv"))
