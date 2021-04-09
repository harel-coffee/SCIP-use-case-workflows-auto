# packages
install.packages("BiocManager")
install.packages("slingshot")
install.packages("devtools")
BiocManager::install("SingleCellExperiment")
devtools::install_github("wesm/feather/R")

# libraries
library(slingshot, quietly = TRUE)
library(feather, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)

df <- read_feather("./notebooks/tmp/data.feather")

exp <- SingleCellExperiment(list(fake=matrix(seq(1, nrow(df)), nrow=1)))
reducedDims(exp) <- list(PCA = data.matrix(df[grep("^feat_pca", names(df))]))
colData(exp)$BGMM <- df$meta_cluster_label

res <- slingshot(
  data = exp,
  clusterLabels = "BGMM",
  reducedDim = "PCA",
  approx_points = 100
)
