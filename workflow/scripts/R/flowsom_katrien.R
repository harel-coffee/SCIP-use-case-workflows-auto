library(FlowSOM, quietly = TRUE)
library(flowCore, quietly = TRUE)
library(arrow, quietly = TRUE)
library(slingshot, quietly = TRUE)
library(SingleCellExperiment, quietly = TRUE)
library(RColorBrewer, quietly = TRUE)

normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

proc <- function(pat) {
  cols <- colnames(exprs)[grep(pat, colnames(exprs))]
  cols_scaled <- sapply(cols, function(x) paste0(x, "_scaled"), USE.NAMES = FALSE)
  cols_mm <- sapply(cols, function(x) paste0(x, "_mm"), USE.NAMES = FALSE)
  
  s <- scale(exprs[, cols])
  m <- normalize(exprs[, cols])
  
  colnames(s) <- cols_scaled
  colnames(m) <- cols_mm
  
  return(list(cols = cols, scaled = s, minmax = m))
}

# read in data and select feature and meta columns

df <- arrow::read_feather("cells.feather")

exprs <- df[grep("^feat", names(df))]
exprs[is.na(exprs)] <- 0
exprs <- exprs[,!apply(exprs, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
exprs <- as.matrix(exprs)

meta <- df[grep("^meta", names(df))]

# fluorescence intensity features
fluor_cols <- colnames(exprs)[grep("^feat_bgcorr.*sum", colnames(exprs))]

# process some morphology features
area <- proc("area")
eccentricity <- proc("eccentricity")
extent <- proc("extent")

# construct flowframe with original + scaled + minmaxed features
ff <- flowFrame(
  exprs = cbind(
    exprs[, c(fluor_cols, area$cols, eccentricity$cols, extent$cols)], 
    area$scaled, eccentricity$scaled, extent$scaled,
    area$minmax, eccentricity$minmax, extent$minmax
  )
)

# logicle transform of fluorescence features
transformList <- flowCore::estimateLogicle(ff, channels = fluor_cols)
ff <- flowWorkspace::transform(ff, transformList)

# min max logicle transformed features
fluor_mm <- normalize(exprs(ff[,fluor_cols]))
colnames(fluor_mm) <- sapply(colnames(fluor_mm), function(x) paste0(x, "_mm"), USE.NAMES = FALSE)

ff <- flowFrame(cbind(exprs(ff), fluor_mm))

# FLOWSOM

nClus <- 15

# fluorescence (logicle transformed) only
fs <- FlowSOM::FlowSOM(ff, compensate = FALSE, scale=FALSE, colsToUse=fluor_cols, xdim=10, ydim=10, nClus=nClus, seed=42)

pdf("output_fluor.pdf")
PlotPies(fs, cellTypes=as.factor(meta$meta_group), backgroundValues=fs$metaclustering)
PlotPies(fs, cellTypes=as.factor(meta$meta_label), backgroundValues=fs$metaclustering)

PlotManualBars(fs, manualVector = as.factor(meta$meta_label))
PlotManualBars(fs, manualVector = as.factor(meta$meta_group))
dev.off()

# minmax fluor (logicle) +  extent + ecc + area
cols <- c(colnames(fluor_mm), colnames(area$minmax), extent$cols, eccentricity$cols)
fs <- FlowSOM::FlowSOM(ff, compensate = FALSE, scale=FALSE, colsToUse=cols, xdim=8, ydim=8, nClus=15, seed=41)

pdf("output_fluor+morph.pdf")
PlotPies(fs, cellTypes=as.factor(meta$meta_group), backgroundValues=fs$metaclustering)
PlotPies(fs, cellTypes=as.factor(meta$meta_label), backgroundValues=fs$metaclustering)

PlotStars(fs, markers=area_cols[grep("DAPI", area_cols)], backgroundValues=fs$metaclustering)

PlotManualBars(fs, manualVector = as.factor(meta$meta_label))
PlotManualBars(fs, manualVector = as.factor(meta$meta_group))
dev.off()
