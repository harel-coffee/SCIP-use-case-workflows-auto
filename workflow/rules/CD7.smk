rule clustering:
    input:
        features="features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
    output:
        "results/{config[use_case]}/figures/cluster_annotation.png"
    conda:
        "environment.yml"
    log:
        notebook="{data}/Leukocyte clustering.ipynb"
    notebook:
        "notebooks/Leukocyte clustering.ipynb"
