library(FlowSOM, quietly = TRUE)
library(flowCore, quietly = TRUE)
library(arrow, quietly = TRUE)
library(slingshot, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer, quietly = TRUE)

df <- arrow::read_feather("cells.feather")

exprs <- df[grep("^feat", names(df))]
exprs[is.na(exprs)] <- 0
exprs <- exprs[,!apply(exprs, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
exprs <- as.matrix(exprs)

meta <- df[grep("^meta", names(df))]

biased_fluor_cols <- colnames(exprs)[grep("^feat_bgcorr_sum.*(Cy5|TMR)", colnames(exprs))]
unbiased_fluor_cols <- colnames(exprs)[grep("^feat_bgcorr_sum_(DAPI)", colnames(exprs))]
labelfree_cols <- colnames(exprs)[grep("^feat_bgcorr_sum_(BF|SSC)", colnames(exprs))]
cols <- colnames(exprs)[!colnames(exprs) %in% c(biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)]
uncorr_cols <- scan("uncorrelated_cols.txt", what="character")
uncorr_cols <- uncorr_cols[!uncorr_cols %in% c(biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)]

normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

ff <- flowFrame(exprs)
trans <- flowCore::estimateLogicle(ff, channels = c("feat_bgcorr_sum_Cy5", "feat_bgcorr_sum_DAPI"))
ff <- flowWorkspace::transform(ff, trans)
trans <- transformList("feat_bgcorr_sum_TMR", logicleTransform(w=1.3))
ff <- flowWorkspace::transform(ff, trans)

ff_mm <- flowFrame(apply(exprs(ff), 2, normalize))

nClus <- 15

# biased fluor only
fsom <- FlowSOM::FlowSOM(ff, compensate = FALSE, scale=FALSE, colsToUse=biased_fluor_cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor_metaclusters.txt", ncolumns=1)

# all intensity
c <- c(biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)
fsom <- FlowSOM::FlowSOM(ff_mm, compensate = FALSE, scale=FALSE, colsToUse=c, xdim=10, ydim=10, nClus=nClus, seed=42)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

# all cols
c <- c(cols, biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)
fsom <- FlowSOM::FlowSOM(ff_mm, compensate = FALSE, scale=FALSE, colsToUse=c, xdim=10, ydim=10, nClus=nClus, seed=42)

PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))

# PCA(all cols)
c <- c(uncorr_cols, biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)
pca <- prcomp(scale(exprs(ff_mm[,c])))
pca_ff <- flowFrame(pca$x)

c <- colnames(pca_ff)
fsom <- FlowSOM::FlowSOM(pca_ff, scale=FALSE, colsToUse=c, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+morphpca_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+morphpca_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+morphpca_metaclusters.txt", ncolumns=1)

# uncorrelated + intensity
c <- c(uncorr_cols, biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)
fsom <- FlowSOM::FlowSOM(ff_mm, scale=FALSE, colsToUse=c, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+uncorr_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+uncorr_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+uncorr_metaclusters.txt", ncolumns=1)

# PCA(uncorr + intensity)
c <- c(uncorr_cols, biased_fluor_cols, unbiased_fluor_cols, labelfree_cols)
pca <- prcomp(scale(exprs(ff_mm[,c])))
pca_ff <- flowFrame(pca$x)

c <- colnames(pca_ff)
fsom <- FlowSOM::FlowSOM(pca_ff, scale=FALSE, colsToUse=c, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+morphpca_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+morphpca_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+morphpca_metaclusters.txt", ncolumns=1)

# PCA(uncorr) + intensity
c <- uncorr_cols
pca <- prcomp(scale(exprs(ff_mm[,c])))
pca_ff <- flowFrame(cbind(apply(pca$x, 2, normalize), exprs(ff_mm[, c(unbiased_fluor_cols, biased_fluor_cols, labelfree_cols)])))

c <- colnames(pca_ff)
fsom <- FlowSOM::FlowSOM(pca_ff, scale=FALSE, colsToUse=c, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+morphpca_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+morphpca_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+morphpca_metaclusters.txt", ncolumns=1)


# slingshot
exp <- SingleCellExperiment(list(fake=matrix(seq(1, nrow(df)), nrow=1)))
reducedDims(exp) <- list(
  PCA = pca$x
)
colData(exp)$cluster <- GetClusters(fs)

res <- slingshot(
  data = exp,
  clusterLabels = "cluster",
  reducedDim = "PCA",
  approx_points = 150,
  start.clus = c(5, 7),
  end.clus = 4
)

embed_res <- embedCurves(
  res,
  reducedDims(exp)$PCA[,1:2],
  approx_points = 150
)

colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
plotcol <- colors[cut(res$slingPseudotime_1, breaks=100)]

plot(reducedDims(exp)$PCA, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(res), lwd=2, col='black')

plot(reducedDims(exp)$PCA, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(res), lwd=2, type = 'lineages', col = 'black')
