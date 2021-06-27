rm(list=list())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)

### -------------------------------------------
n <- 1000
e <- rnorm(n,0, 0.1)

X    <- c(1, 1)
phi1 <- 0.5
phi2 <- -0.5
c    <- 1
t    <- 3
for(t in 2:n){
  X[t] <- c + phi1 * X[t-1] + e[t]
  # print(X[t])
}

plot(X, type='l')

acf(diff(X))




### -----------------------
library(marima)
df <- read_csv("data_pump_30_small_cleaned.csv")[3:8]

model <- define.model(kvar=6, ar=c(1, 2), ma=0, indep=NULL)
marima_model <- marima(df, means=0, ar.pattern = model$ar.pattern, 
                       ma.pattern = model$ma.pattern, Check = F, Plot = "none",
                       penalty = 0)
marima_model$averages

n <- 1000
marima_sim <- marima.sim(kvar=6, ar.model = marima_model$ar.estimates,
                         ma.model = marima_model$ma.estimates, seed=1234,
                         averages = marima_model$averages,
                         resid.cov = marima_model$resid.cov,
                         nsim = n)
cor(marima_sim)

ts <- as.ts(marima_sim)
plot(ts)
