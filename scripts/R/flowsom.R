library(FlowSOM, quietly = TRUE)
library(flowCore, quietly = TRUE)
library(arrow, quietly = TRUE)
library(slingshot, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer, quietly = TRUE)
library(ggplot2, quietly = TRUE)
library(tidyr)
library(dplyr)
source('/data/dev/active/analysis/weizmann-ehv-analysis/scripts/R/PlotDimRed.R')

pca <- function(mat) {
  mat.pr <- prcomp(mat, center = TRUE, scale = TRUE)
  cumpro <- cumsum(mat.pr$sdev^2 / sum(mat.pr$sdev^2))
  pc_keep <- sum(cumpro < 0.95)+1 # explain 95% variance
  return(mat.pr$x[, colnames(mat.pr$x)[0:pc_keep]])
}

minmax <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

quant_quant <- function(x, na.rm = TRUE, probs = c(0.01, 0.99)) {
  qq <- quantile(x, probs)
  return((x - qq[1]) /(qq[2] - qq[1]))
}

zscore <- function(x, na.rm = TRUE) {
  return (x - mean(x)) / std(x)
}

robust <- function(x, na.rm = TRUE) {
  return (x - median(x)) / mad(x)
}

plot_clusters <- function(ff, fsom) {
  cols <- all_of(colnames(ff))
  
  cells_long <- pivot_longer(as.data.frame(exprs(ff)), 
                             cols = cols,
                             names_to = "Feature", values_to = "Value")
  clusters_long <- pivot_longer(as.data.frame(GetClusterMFIs(fsom)), 
                                cols = cols,
                                names_to = "Feature", values_to = "Value") 
  metaclusters_long <- pivot_longer(GetMetaclusterMFIs(fsom), 
                                    cols = cols,
                                    names_to = "Feature", values_to = "Value") 
  
  colors <- c("Cells" = "#fdcc8a", "Clusters" = "black", "Metaclusters" = "#e34a33")
  cells_long%>% 
    ggplot(aes(x = Feature, y = Value)) +
    geom_violin(aes(color = "Cells", fill = "Cells")) +
    ggbeeswarm::geom_quasirandom(aes(color = "Clusters"), data = clusters_long, size = 0.5) +
    ggbeeswarm::geom_quasirandom(aes(color = "Metaclusters"), data = metaclusters_long) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90)) +
    scale_color_manual(values = colors, name = "")+
    scale_fill_manual(values = colors, name = "") +
    guides(fill = FALSE)
}

do_flowsom <- function(ff, nClus, nDim, out) {
  
  out = sprintf("%s_%d_%d", out, nClus, nDim)
  
  fsom <- FlowSOM::FlowSOM(ff, compensate = FALSE, scale=FALSE, xdim=nDim, ydim=nDim, nClus=nClus, seed=42)
  
  pdf(sprintf("fsom/%s.pdf", out))
  
  print(PlotPies(fsom, cellTypes=as.factor(meta$meta_group), backgroundValues=fsom$metaclustering))
  print(PlotPies(fsom, cellTypes=as.factor(meta$meta_label), backgroundValues=fsom$metaclustering))
  
  print(PlotDimRed(fsom, dimred=umap, colorBy = "metaclusters"))
  print(PlotDimRed(fsom, dimred=umap, colorBy = "clusters"))
  
  data <- cbind(umap, meta$meta_label)
  colnames(data) <- c("umap_0", "umap_1", "label")
  p <- ggplot2::ggplot(data) +
    scattermore::geom_scattermore(ggplot2::aes(x = .data$umap_0,
                                               y = .data$umap_1,
                                               col = .data$label),
                                  pointsize = 1) +
    ggplot2::theme_minimal() +
    ggplot2::coord_fixed()
  print(p)
  
  PlotManualBars(fsom, manualVector = as.factor(meta$meta_group))
  PlotManualBars(fsom, manualVector = as.factor(meta$meta_label))
  
  dev.off()
  
  return(fsom)
}

ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}


## COMPARISON OF SCALINGS

df <- arrow::read_feather("cells.feather")
df2 <- arrow::read_feather("slingshot.feather")

meta <- df[grep("^meta", names(df))]
umap <- df2[grep("^umap", names(df2))]

exprs <- df[grep("^feat", names(df))]
exprs[is.na(exprs)] <- 0

# remove zero variance columns
exprs <- exprs[,!apply(exprs, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]

exprs <- as.matrix(exprs)

# logicle transform of sum features
ff <- flowFrame(exprs)
log_cols <- colnames(ff)[c(grep("sum", colnames(ff)), grep("max", colnames(ff)))]
transformList <- flowCore::estimateLogicle(ff, channels = log_cols)
ff <- flowWorkspace::transform(ff, transformList)

exprs <- exprs(ff)

X <- list(
  exprs,
  apply(exprs, 2, minmax),
  apply(exprs, 2, quant_quant, probs = c(0.01, 0.99)),
  apply(exprs, 2, quant_quant, probs = c(0.05, 0.95)),
  apply(exprs, 2, zscore),
  apply(exprs, 2, robust),
  pca(exprs),
  pca(apply(exprs, 2, minmax)),
  pca(apply(exprs, 2, quant_quant, probs = c(0.01, 0.99))),
  pca(apply(exprs, 2, quant_quant, probs = c(0.05, 0.95))),
  pca(apply(exprs, 2, zscore)),
  pca(apply(exprs, 2, robust))
)

nClus <- 4
nDim <- 10

ids <- c(
  "noop",
  "minmax",
  "qq1-99",
  "qq5-95",
  "zscore",
  "robust",
  "pca_noop",
  "pca_minmax",
  "pca_qq1-99",
  "pca_qq5-95",
  "pca_zscore",
  "pca_robust"
)

fsoms <- c()
for (i in 1:length(ids)) {
  print(ids[i])
  fsoms[[i]] <- do_flowsom(flowFrame(X[[i]]), nClus, nDim, ids[i])
}

purities <- unlist(map(fsoms, function(x) {ClusterPurity(GetMetaclusters(x), meta$meta_label)}))

id <- 2
ff <- flowFrame(X[[id]])
plot_clusters(ff, fsoms[[id]])


## MINMAX + ISO preproc
df <- arrow::read_feather("cells_scaled.feather")
df2 <- arrow::read_feather("slingshot.feather")

meta <- df[grep("^meta", names(df))]
umap <- df2[grep("^umap", names(df2))]

exprs <- df[grep("^feat", names(df))]
exprs <- as.matrix(exprs)

ff <- flowFrame(exprs)
fsom <- do_flowsom(ff, nClus = 20, nDim = 10, out ="mm_iso_pre")
plot_clusters(ff, fsom)
ClusterPurity(GetMetaclusters(fsom), meta$meta_label)
write(GetMetaclusters(fsom), file="fsom/metaclusters_mm_iso_pre_20_10.txt", ncolumns=1)

ff <- flowFrame(pca(exprs))
fsom <- do_flowsom(ff, nClus = 20, nDim = 10, out ="mm_iso_pre_pca")
plot_clusters(ff, fsom)
ClusterPurity(GetMetaclusters(fsom), meta$meta_label)
