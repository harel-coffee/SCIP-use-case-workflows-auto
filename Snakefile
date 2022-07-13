
rule preprocessing:
    input:
        files=expand(
            "{data}/features.{part}.parquet",
            part=range(5),
            allow_missing=True,
        ),
        data_dir="{data}"
    output:
        "{data}/features.parquet"
    conda:
        "environment.yml"
    log:
        notebook="{data}/notebooks/QC/processing_scip_features.ipynb"
    notebook:
        "notebooks/preprocessing/{config[set]}_processing_scip_features.ipynb"


rule quality_control:
    input:
        "{data}/features.parquet",
    output:
        "{data}/indices/columns.npy",
        "{data}/indices/index.npy",
    log:
        notebook="{data}/notebooks/QC/quality_control.ipynb"
    notebook:
        "notebooks/QC/{config[set]}_quality_control.ipynb"

rule hyperparameter_optimization:
    input:
        features="{data}/features.parquet",
        columns="{data}/indices/columns.npy",
        index="{data}/indices/index.npy"
    output:
        "{data}/hpo/{grid}_{full}.pickle",
    params:
        set=config["set"],
        grid="{grid}"
    threads:
        10
    log:
        "{data}/log/hyperparameter_optimization_{grid}_{full}.log"
    conda:
        "environment.yml"
    script:
        "scripts/python/{params.set}_xgb_parameter_search.py"

rule WBC_IFC_classification:
    input:
        features="{data}/features.parquet",
        columns="{data}/indices/columns.npy",
        index="{data}/indices/index.npy",
        hpo_grid="{data}/grid/rsh.pickle",
    output:
        "{data}/models/xgb.pickle",
    log:
        notebook="{data}/notebooks/QC/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"
    notebook:
        "notebooks/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"
