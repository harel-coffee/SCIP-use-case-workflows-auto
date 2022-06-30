
rule preprocessing:
    input:
        files=expand(
            "{data}/scip/{run}/features.{part}.parquet",
            part=range(5),
            allow_missing=True,
        ),
        data_dir="{data}/scip/{run}"
    output:
        "{data}/scip/{run}/features.parquet",
    conda:
        "environment.yml"
    log:
        notebook="{data}/scip/{run}/notebooks/QC/processing_scip_features.ipynb"
    notebook:
        "notebooks/preprocessing/{config[set]}_processing_scip_features.ipynb"


rule quality_control:
    input:
        "{data}/scip/{run}/features.parquet",
    output:
        "{data}/scip/{run}/indices/columns.npy",
        "{data}/scip/{run}/indices/index.npy",
    log:
        notebook="{data}/scip/{run}/notebooks/QC/{config[set]}_quality_control.ipynb"
    notebook:
        "notebooks/QC/{config[set]}_quality_control.ipynb"


rule hyperparameter_optimization:
    input:
        "{data}/scip/{run}/features.parquet",
    output:
        "{data}/scip/{run}/grid/rsh.pickle",
    script:
        "scripts/python/{config[set]}_xgb_parameter_search.py"


rule WBC_IFC_SCIP:
    input:
        path="{data}/images/",
        config="{data}/scip.yml",
    output:
        expand(
            "{data}/scip/{run}/features.{part}.parquet",
            part=range(5),
            allow_missing=True,
        ),
    shell:
        "scip --mode mpi ... {input.config} {wildcards.data}/scip/{wildcards.run} {input.path}"


rule WBC_IFC_classification:
    input:
        features="{data}/scip/{run}/features.parquet",
        columns="{data}/scip/{run}/indices/columns.npy",
        index="{data}/scip/{run}/indices/index.npy",
        hpo_grid="{data}/scip/{run}/grid/rsh.pickle",
    output:
        "{data}/scip/{run}/models/xgb.pickle",
    log:
        notebook="{data}/scip/{run}/notebooks/QC/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"
    notebook:
        "notebooks/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"
