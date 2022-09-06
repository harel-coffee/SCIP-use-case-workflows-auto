rule BBBC021_moa_prediction:
    input:
        features="features.parquet",
        moa="BBBC021_v1_moa.csv",
        image="BBBC021_v1_image.csv"
    output:
        confusion_matrix="figures/confusion_matrix.png"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/moa_prediction.ipynb"
    notebook:
        "../notebooks/moa_prediction.ipynb"