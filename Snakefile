rule preprocessing:
    input:
        expand(
            "{data}/features.{part}.parquet",
            part=range(int(config['parts'])),
            allow_missing=True
        )
    output:
        "{data}/features.parquet"
    threads: 1
    conda:
        "environment.yml"
    log:
        notebook="{data}/notebooks/preprocessing/processing_scip_features.ipynb"
    notebook:
        "notebooks/preprocessing/{config[set]}_processing_scip_features.ipynb"

rule labels:
    output:
        "{data_root}/scip/{data_postfix}/labels.parquet"
    conda:
        "environment.yml"
    threads: 1
    log:
        notebook="{data_root}/scip/{data_postfix}/notebooks/preprocessing/labels.ipynb"
    notebook:
        "notebooks/preprocessing/config[set]_labels.ipynb"
        
use rule labels as WBC_IFC_labels with:
    input:
        features="{data_root}/scip/{data_postfix}/features.parquet",
        population_dir="{data_root}/meta/"
        
use rule labels as BBBC_labels with:
    input:
        image_path="{data_root}/BBBC021_v1_image.csv",
        output="{data_root}/labels.parquet"

rule quality_control:
    input:
        "{data}/features.parquet"
    output:
        columns="{data}/indices/columns.npy",
        index="{data}/indices/index.npy"
    conda:
        "environment.yml"
    log:
        notebook="{data}/notebooks/QC/quality_control.ipynb"
    notebook:
        "notebooks/QC/{config[set]}_quality_control.ipynb"


rule all_hyperparameter_optimization:
    input:
        expand(
            "{data}/hpo/{grid}_{full}.pickle",
            full=["full", "cyto"],
            grid=["rsh", "random"],
            data=config["data"]
        )


rule hyperparameter_optimization:
    input:
        features="{data_root}/{type}/{date}/features.parquet",
        columns="{data_root}/{type}/{date}/indices/columns.npy",
        index="{data_root}/{type}/{date}/indices/index.npy",
        labels="{data_root}/{type}/{date}/labels.parquet"
    output:
        "{data_root}/{type}/{date}/hpo/{grid}_{full}.pickle"
    conda:
        "environment.yml"
    params:
        set=config["set"],
        grid="{grid}"
    threads:
        10
    log:
        "{data_root}/{type}/{date}/log/hyperparameter_optimization_{grid}_{full}.log"
    conda:
        "environment.yml"
    script:
        "scripts/python/{params.set}_xgb_parameter_search.py"


rule WBC_IFC_classification:
    input:
        features="{data}/features.parquet",
        columns="{data}/indices/columns.npy",
        index="{data}/indices/index.npy",
        hpo_grid="{data}/grid/rsh.pickle"
    output:
        "{data}/models/xgb.pickle"
    conda:
        "environment.yml"
    log:
        notebook="{data}/notebooks/Stain-free Leukocyte Prediction.ipynb"
    notebook:
        "notebooks/Stain-free Leukocyte Prediction.ipynb"

rule WBC_CD7_clustering:
    input:
        features="{data}/features.parquet",
        columns="{data}/indices/columns.npy",
        index="{data}/indices/index.npy",
    output:
        "{data}/figures/cluster_annotation.png"
    conda:
        "environment.yml"
    log:
        notebook="{data}/Leukocyte clustering.ipynb"
    notebook:
        "notebooks/Leukocyte clustering.ipynb"

