PlotDimRed <- function(fsom,
                       colsToUse = fsom$map$colsUsed,
                       colorBy = "metaclusters",
                       cTotal = NULL,
                       dimred = Rtsne::Rtsne,
                       extractLayout  = function(dimred){dimred$Y},
                       label = TRUE,
                       returnLayout = FALSE,
                       seed = NULL,
                       title = NULL,
                       ...){
  dimred_data <- fsom$data
  if (length(colorBy) == 1 && colorBy == "metaclusters") {
    if (is.null(fsom$metaclustering)) stop("No metaclustering present")
    dimred_col <- as.data.frame(GetMetaclusters(fsom))
  } else if (length(colorBy) == 1 && colorBy == "clusters") {
    dimred_col <- as.data.frame(as.factor(GetClusters(fsom)))
  } else if (all(colorBy %in% colnames(dimred_data)) ||
             all(colorBy %in% GetMarkers(fsom, colnames(dimred_data))) ||
             all(colorBy %in% seq_len(ncol(dimred_data)))){
    dimred_col <- fsom$data[, GetChannels(fsom, colorBy), drop = FALSE]
    colnames(dimred_col) <- GetMarkers(fsom, colnames(dimred_col))
    colorBy <- "marker"
  } else stop(paste0("colorBy should be \"metaclusters\", \"clusters\" or a ",
                     "vector of channels, markers or indices"))
  if (!is.null(colsToUse)) dimred_data <- dimred_data[, GetChannels(fsom, 
                                                                    colsToUse)]
  if (!is.null(seed)) set.seed(seed)
  if (!is.null(cTotal) && cTotal < nrow(dimred_data)) {
    downsample <- sample(seq_len(nrow(dimred_data)), cTotal)
    dimred_data <- dimred_data[downsample, , drop = FALSE]
    dimred_col <- dimred_col[downsample, , drop = FALSE]
  } else {
    downsample <- seq_len(nrow(dimred_data))
  }
  if (is.function(dimred)){
    dimred_res <- dimred(dimred_data, ...)
    dimred_layout <- as.data.frame(extractLayout(dimred_res))
    if (nrow(dimred_layout) == 0 && ncol(dimred_layout) != 2) {
      stop("Please use the right extraction function in extractLayout")
    }
  } else if((is.matrix(dimred) | is.data.frame(dimred)) & 
            (nrow(dimred) == nrow(dimred_data) | 
             any(colnames(dimred) == "Original_ID"))){
    dimred_layout <- as.data.frame(dimred)
    if("Original_ID" %in% colnames(dimred)){
      id_col <- which(colnames(dimred) == "Original_ID")
      dimred_layout <- dimred_layout[,-id_col]
      dimred_data <- dimred_data[dimred[,"Original_ID"], , drop = FALSE]
      dimred_col <- dimred_col[dimred[,"Original_ID"], , drop = FALSE]
    }
  } else stop("dimred should be a dimensionality reduction method or matrix")
  
  colnames(dimred_layout) <- c("dimred_1", "dimred_2")
  dimred_plot <- cbind(dimred_layout, dimred_col)
  
  if (colorBy == "marker"){
    dimred_plot <- dimred_plot %>% tidyr::pivot_longer(3:ncol(dimred_plot),
                                                       names_to = "markers")
    p <- ggplot2::ggplot(dimred_plot) +
      scattermore::geom_scattermore(ggplot2::aes(x = .data$dimred_1,
                                                 y = .data$dimred_2,
                                                 col = .data$value),
                                    pointsize = 1) +
      ggplot2::facet_wrap(~markers) +
      ggplot2::theme_minimal() +
      ggplot2::coord_fixed() +
      ggplot2::scale_color_gradientn(colors = FlowSOM_colors(9))
  } else {
    colnames(dimred_plot) <- c("dimred_1", "dimred_2", "colors")
    
    median_x <- tapply(dimred_plot[,"dimred_1"], dimred_plot[,"colors"], median)
    median_y <- tapply(dimred_plot[,"dimred_2"], dimred_plot[,"colors"], median)
    
    p <- ggplot2::ggplot(dimred_plot) +
      scattermore::geom_scattermore(ggplot2::aes(x = .data$dimred_1,
                                                 y = .data$dimred_2,
                                                 col = .data$colors),
                                    pointsize = 1) +
      ggplot2::theme_minimal() +
      ggplot2::coord_fixed()
    if (label){
      p <- p + ggrepel::geom_label_repel(aes(x = .data$x,
                                             y = .data$y,
                                             label = .data$label,
                                             color = .data$label),
                                         data = data.frame(x = median_x,
                                                           y = median_y,
                                                           label = names(median_x)),
                                         segment.color = "gray", force = 20,
                                         segment.size = 0.2, point.padding = 0.5)+
        labs(col = colorBy)
    }
  }
  if (!is.null(title)) p <- p + ggplot2::ggtitle(title)
  if (returnLayout) {
    if (!is.null(cTotal)){
      dimred_layout <- data.frame(dimred_layout, "Original_ID" = downsample)
    }
    return(list("plot" = p, "layout" = dimred_layout))
  } else return(p)
}