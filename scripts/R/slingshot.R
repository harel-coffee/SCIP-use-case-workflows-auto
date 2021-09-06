# packages
install.packages("BiocManager")
install.packages("devtools")
BiocManager::install("SingleCellExperiment")
devtools::install_github("wesm/feather/R")

BiocManager::install(version='devel')
BiocManager::install("slingshot")

# libraries
library(slingshot, quietly = TRUE)
library(feather, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer)

df <- read_feather("./notebooks/tmp/data.feather")

exp <- SingleCellExperiment(list(fake=matrix(seq(1, nrow(df)), nrow=1)))
reducedDims(exp) <- list(
  PCA = data.matrix(df[grep("^feat_pca_[0-9]$", names(df))]),
  UMAP = data.matrix(df[grep("^feat_umap", names(df))])
)
colData(exp)$BGMM <- df$meta_cluster_label

res <- slingshot(
  data = exp,
  clusterLabels = "BGMM",
  reducedDim = "PCA",
  approx_points = 500
)

colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
plotcol <- colors[cut(res$slingPseudotime_1, breaks=100)]

plot(reducedDims(exp)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(res), lwd=2, col='black')

plot(reducedDims(exp)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(res), lwd=2, type = 'lineages', col = 'black')
