# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/2c_selected_features_correlation_analysis.ipynb (unless otherwise specified).

__all__ = ['get_correlated_features']

# Cell
def get_correlated_features(df, thresh=0.8):
    correlation = df.corr()
    var = df.var()
    correlated_features = set()

    # loop over all feature combinations
    for i in range(correlation.shape[0]):
        for j in range(i):

            # check absolute value of correlation against threshold
            if abs(correlation.iloc[i, j]) > thresh:

                # keep feature with most variance
                if var[correlation.columns[i]] >= var[correlation.columns[j]]:
                    colname = correlation.columns[j]
                else:
                    colname = correlation.columns[i]
                correlated_features.add(colname)

    return correlated_features