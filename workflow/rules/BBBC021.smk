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

rule BBBC021_comparison:
    input:
        features="features.parquet",
        moa="BBBC021_v1_moa.csv",
        image="BBBC021_v1_image.csv",
        gt="BBBC021_v1.sqlite"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/comparison_to_cellprofiler.ipynb"
    notebook:
        "../notebooks/BBBC021/comparison_to_cellprofiler.ipynb"