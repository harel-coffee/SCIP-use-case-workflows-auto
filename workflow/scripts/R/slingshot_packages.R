# packages
install.packages("BiocManager")
install.packages("devtools")
BiocManager::install("SingleCellExperiment")
devtools::install_github("wesm/feather/R")

BiocManager::install(version='devel')
BiocManager::install("slingshot")