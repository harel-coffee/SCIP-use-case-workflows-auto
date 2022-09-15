rule clustering:
    input:
        features="features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
    output:
        "figures/cluster_annotation.png"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/clustering.ipynb"
    notebook:
        "../notebooks/clustering.ipynb"
        
rule image_inspection:
    input:
        "Experiment-800.czi"
    output:
        scenes.txt
    log:
        notebook="notebooks/image_inspection.ipynb"
    notebook:
        "../notebooks/image_inspection.ipynb"
