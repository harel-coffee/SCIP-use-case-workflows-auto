# libraries
library(slingshot, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer)
library(arrow, quietly = TRUE)

df <- arrow::read_feather("slingshot.feather")
cluster <- read.delim("fsom/metaclusters_mm_iso_pre_20_10.txt", sep="\n", header=FALSE)[[1]]

exp <- SingleCellExperiment(list(fake=matrix(seq(1, nrow(df)), nrow=1)))
reducedDims(exp) <- list(
  PCA = data.matrix(df[grep("^pca", names(df))]),
  UMAP = data.matrix(df[grep("^umap", names(df))])
)
colData(exp)$cluster <- cluster

exp <- SingleCellExperiment(list(fake=matrix(seq(1, nrow(df2)), nrow=1)))
reducedDims(exp) <- list(
  PCA = data.matrix(df2[grep("^pca", names(df2))]),
  UMAP = data.matrix(df2[grep("^umap", names(df2))])
)
colData(exp)$cluster <- cluster

res <- slingshot(
  data = exp,
  clusterLabels = "cluster",
  reducedDim = "PCA",
  approx_points = 10,
  stretch = 0,
  start.clus = 11
)

embed_res <- embedCurves(
  res,
  reducedDims(exp)$UMAP,
  approx_points = 10,
  stretch = 0,
  start.clus = 11
)

nc <- 3
pt <- slingPseudotime(embed_res)
nms <- colnames(pt)
nr <- ceiling(length(nms)/nc)
pal <- viridis(100, end = 0.95)
par(mfrow = c(nr, nc))
for (i in nms) {
  colors <- pal[cut(pt[,i], breaks = 100)]
  plot(reducedDims(exp)$UMAP, col = colors, pch = 16, cex = 0.5, main = i)
  lines(SlingshotDataSet(embed_res), lwd = 2, col = 'black', type = 'lineages')
}
