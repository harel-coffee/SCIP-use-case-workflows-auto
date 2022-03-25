# libraries
library(slingshot, quietly = TRUE)
library(feather, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer)

df <- read_feather("slingshot.feather")

exp <- SingleCellExperiment(list(fake=matrix(seq(1, nrow(df)), nrow=1)))
reducedDims(exp) <- list(
  PCA = data.matrix(df[grep("^pca", names(df))]),
  UMAP = data.matrix(df[grep("^umap", names(df))])
)
colData(exp)$GMM <- df$cluster

res <- slingshot(
  data = exp,
  clusterLabels = "GMM",
  reducedDim = "PCA",
  approx_points = 500
)

pt <- as.data.frame(res$slingPseudotime_1)
write_feather(pt, 'pt.feather')

colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
plotcol <- colors[cut(res$slingPseudotime_1, breaks=100)]

plot(reducedDims(exp)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(res), lwd=2, col='black')

plot(reducedDims(exp)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(res), lwd=2, type = 'lineages', col = 'black')
