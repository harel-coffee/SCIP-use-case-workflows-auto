rule preprocessing:
    input:
        files=expand(
            "{data}/features.{part}.parquet",
            part=range(10),
            allow_missing=True,
        ),
        data_dir="{data}"
    output:
        "{data}/features.parquet"
    conda:
        "environment.yml"
    log:
        notebook="{data}/notebooks/preprocessing/processing_scip_features.ipynb"
    notebook:
        "notebooks/preprocessing/{config[set]}_processing_scip_features.ipynb"


# rule WBC_IFC_preprocessing:
#     input:
#         files=expand(
#             "{data_root}/{data_postfix}/features.{part}.parquet",
#             part=range(10),
#             allow_missing=True,
#         ),
#         data_dir="{data_root}/{data_postfix}",
#     output:
#         "{data_root}/{data_postfix}/features.parquet"
#     conda:
#         "environment.yml"
#     log:
#         notebook="{data_root}/{data_postfix}/notebooks/preprocessing/processing_scip_features.ipynb"
#     notebook:
#         "notebooks/preprocessing/wbc_processing_scip_features.ipynb"


rule WBC_IFC_labels:
    input:
        features="{data_root}/scip/{data_postfix}/features.parquet",
        population_dir="{data_root}/meta/"
    output:
        "{data_root}/scip/{data_postfix}/labels.parquet"
    conda:
        "environment.yml"
    log:
        notebook="{data_root}/scip/{data_postfix}/notebooks/preprocessing/wbc_labels.ipynb"
    notebook:
        "notebooks/preprocessing/wbc_labels.ipynb"


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


rule hyperparameter_optimization:
    input:
        features="{data}/features.parquet",
        columns="{data}/indices/columns.npy",
        index="{data}/indices/index.npy",
        labels="{data}/labels.parquet"
    output:
        "{data}/hpo/{grid}_{full}.pickle"
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
        hpo_grid="{data}/grid/rsh.pickle"
    output:
        "{data}/models/xgb.pickle"
    conda:
        "environment.yml"
    log:
        notebook="{data}/notebooks/QC/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"
    notebook:
        "notebooks/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"
