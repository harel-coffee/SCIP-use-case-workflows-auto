rule all_clustering:
    input:
        "adata_0.h5ad", "adata_1.h5ad"

rule clustering:
    input:
        features="features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy"
    output:
        "adata_{fillna}.h5ad"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/clustering_{fillna}.ipynb"
    notebook:
        "../notebooks/CD7/clustering.ipynb"
        
rule cluster_annotation:
    input:
        "adata_0.h5ad"
    output:
        "figures/cluster_annotation.png"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/cluster_annotation.ipynb"
    notebook:
        "../notebooks/CD7/cluster_annotation.ipynb"
        
rule image_inspection:
    input:
        "Experiment-800.czi"
    output:
        "scenes.txt"
    log:
        notebook="notebooks/image_inspection.ipynb"
    notebook:
        "../notebooks/image_inspection.ipynb"
