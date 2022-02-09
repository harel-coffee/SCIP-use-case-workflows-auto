# libraries
library(slingshot, quietly = TRUE)
library(feather, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer)

df <- read_feather('/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_skitty/slingshot_pca0.75.feather')

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
  approx_points = 150,
  omega = TRUE, start.clus = c(1,11)
)

pt <- as.data.frame(res$slingPseudotime_1)
write_feather(pt, '/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_skitty/pt_pca0.75.feather')

embed_res <- embedCurves(
  res,
  reducedDims(exp)$UMAP,
  approx_points = 150
)

pdf("/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_skitty/slingshot_umap_pca0.75.pdf")
colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
plotcol <- colors[cut(res$slingPseudotime_1, breaks=100)]

plot(reducedDims(exp)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(embed_res), lwd=2, col='black')

plot(reducedDims(exp)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(embed_res), lwd=2, type = 'lineages', col = 'black')

dev.off()

embed_res <- embedCurves(
  res,
  reducedDims(exp)$PCA[,1:2],
  approx_points = 150
)

pdf("/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_skitty/slingshot_pca_pca0.75.pdf")
colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
plotcol <- colors[cut(res$slingPseudotime_1, breaks=100)]

plot(reducedDims(exp)$PCA[,1:2], col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(embed_res), lwd=2, col='black')

plot(reducedDims(exp)$PCA[,1:2], col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(embed_res), lwd=2, type = 'lineages', col = 'black')

dev.off()
