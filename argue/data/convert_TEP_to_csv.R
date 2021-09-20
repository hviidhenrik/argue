# init
rm(list=ls()); setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)

# load phase 1 and 2 data from the data folder in the project root
path <- "..\\..\\data\\TEP\\"
load(paste0(path, "TEP_FaultFree_Testing.RData"))
load(paste0(path, "TEP_Faulty_Testing.RData"))
df_phase1         <- tibble(fault_free_testing)
df_phase2         <- tibble(faulty_testing)

# make reduced datasets for faster development iteration
df_phase1_reduced <- df_phase1 %>% filter(simulationRun %in% seq(1:5))
df_phase2_reduced <- df_phase2 %>% filter(simulationRun %in% seq(1:5))

# assign new binary column "faulty". Phase 2 gets faulty after sample 160
df_phase1_reduced$faulty             = 0
df_phase1_reduced                   <- df_phase1_reduced[, c(1,2,3,56,4:55)]

df_phase2_reduced$faulty             = 0
df_phase2_reduced                   <- df_phase2_reduced[, c(1,2,3,56,4:55)]
idx_faulty                          <- which(df_phase2_reduced$sample > 159)
df_phase2_reduced$faulty[idx_faulty] = 1

# clean up workspace for memory saving
rm(list=c("df_phase1", "df_phase2", "fault_free_testing", "faulty_testing"))

# save data to csv
df_phase1_reduced %>% write_csv(file = paste0(path, "data_tep_phase1.csv"))
df_phase2_reduced %>% write_csv(file = paste0(path, "data_tep_phase2.csv"))

