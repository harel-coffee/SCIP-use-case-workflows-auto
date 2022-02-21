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

fluor_cols <- colnames(exprs)[grep("^feat_bgcorr.*sum", colnames(exprs))]
uncorr_cols <- scan("uncorrelated_cols.txt", what="character")

normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

proc <- function(cols) {
  cols_scaled <- sapply(cols, function(x) paste0(x, "_scaled"), USE.NAMES = FALSE)
  cols_mm <- sapply(cols, function(x) paste0(x, "_mm"), USE.NAMES = FALSE)
  
  s <- scale(exprs[, cols])
  m <- sapply(cols, function(x) normalize(exprs[, x]))
  
  colnames(s) <- cols_scaled
  colnames(m) <- cols_mm
  
  return(list(cols = cols, scaled = s, minmax = m))
}

area <- proc(colnames(exprs)[grep("area", colnames(exprs))])
eccentricity <- proc(colnames(exprs)[grep("eccentricity", colnames(exprs))])
extent <- proc(colnames(exprs)[grep("extent", colnames(exprs))])
majoraxis <- proc(colnames(exprs)[grep("major_axis", colnames(exprs))])
minoraxis <- proc(colnames(exprs)[grep("minor_axis", colnames(exprs))])
diameter <- proc(colnames(exprs)[grep("diameter", colnames(exprs))])
uncorr <- proc(uncorr_cols)

ff <- flowFrame(
  exprs = cbind(
    exprs[, c(fluor_cols, area$cols, eccentricity$cols, extent$cols)], 
    area$scaled, eccentricity$scaled, extent$scaled, majoraxis$scaled, minoraxis$scaled, diameter$scaled,
    area$minmax, majoraxis$minmax, minoraxis$minmax, diameter$minmax
  )
)
transformList <- flowCore::estimateLogicle(ff, channels = fluor_cols)
ff <- flowWorkspace::transform(ff, transformList)

fluor_mm <- normalize(exprs(ff[,fluor_cols]))
colnames(fluor_mm) <- sapply(colnames(fluor_mm), function(x) paste0(x, "_mm"), USE.NAMES = FALSE)

ff <- flowFrame(cbind(exprs(ff), fluor_mm))
ff_uncorr <- flowFrame(cbind(exprs[, uncorr_cols], uncorr$minmax, fluor_mm))

nClus <- 15

# fluor only
fsom <- FlowSOM::FlowSOM(ff, compensate = FALSE, scale=FALSE, colsToUse=fluor_cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor_metaclusters.txt", ncolumns=1)

# minmax fluor + ecc + minmax area + major + minor + diameter
cols <- c(
  colnames(fluor_mm), 
  colnames(area$minmax), 
  colnames(majoraxis$minmax),
  colnames(minoraxis$minmax),
  colnames(diameter$minmax),
  eccentricity$cols
)
fsom <- FlowSOM::FlowSOM(ff, compensate = FALSE, scale=FALSE, colsToUse=cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+morph_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+morph_sommary.pdf")

# PCA
cols <- c(
  colnames(area$minmax), 
  colnames(majoraxis$minmax),
  colnames(minoraxis$minmax),
  colnames(diameter$minmax),
  extent$cols,
  eccentricity$cols
)
pca <- prcomp(scale(exprs(ff[,cols])), rank=10)
pca_ff <- flowFrame(cbind(apply(pca$x, 2, normalize), exprs(ff[,colnames(fluor_mm)])))

cols <- colnames(pca_ff)
fsom <- FlowSOM::FlowSOM(pca_ff, scale=FALSE, colsToUse=cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+morphpca_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+morphpca_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+morphpca_metaclusters.txt", ncolumns=1)

# uncorrelated
cols <- c(colnames(fluor_mm), colnames(uncorr$minmax))
fsom <- FlowSOM::FlowSOM(ff_uncorr, scale=FALSE, colsToUse=cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+uncorr_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+uncorr_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+uncorr_metaclusters.txt", ncolumns=1)

# uncorrelated + pca
cols <- uncorr$cols
pca <- prcomp(scale(exprs(ff_uncorr[,cols])))
pca_ff <- flowFrame(cbind(apply(pca$x, 2, normalize), exprs(ff[,colnames(fluor_mm)])))
cols <- colnames(pca_ff)
fsom <- FlowSOM::FlowSOM(pca_ff, scale=FALSE, colsToUse=cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("fluor+uncorr_pies.pdf")
PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering)
PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering)

PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
dev.off()

FlowSOMmary(fsom, plotFile = "fluor+uncorr_sommary.pdf")

write(GetMetaclusters(fsom), file="fluor+uncorr_metaclusters.txt", ncolumns=1)

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
