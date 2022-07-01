rule preprocessing:
    input:
        expand("{data}/scip/{run}/features.{part}.parquet", part=range(10), allow_missing=True)
    output:
        "{data}/scip/{run}/features.parquet"
    script:
        "scripts/python/preprocessing.py {input}"

rule quality_control:
    input:
        "{data}/scip/{run}/features.parquet"
    output:
        "{data}/scip/{run}/indices/columns.npy",
        "{data}/scip/{run}/indices/index.npy"
    notebook:
        "notebooks/QC/{config[set]}_quality_control.ipynb"

rule hyperparameter_optimization:
    input:
        features="{data}/scip/{run}/features.parquet"
        columns="{data}/scip/{run}/indices/columns.npy"
        index="{data}/scip/{run}/indices/index.npy"
    output:
        "{data}/scip/{run}/grid/rsh.pickle"
    script:
        "scripts/python/{config[set]}_xgb_parameter_search.py"

rule WBC_IFC_SCIP:
    input:
        path="{data}/images/",
        config="{data}/scip.yml"
    output:
        expand("{data}/scip/{run}/features.{part}.parquet", part=range(10), allow_missing=True),
    shell:
        "scip --mode mpi ... {input.config} {wildcards.data}/scip/{wildcards.run} {input.path}"

rule WBC_IFC_classification:
    input:
        features="{data}/scip/{run}/features.parquet",
        columns="{data}/scip/{run}/indices/columns.npy",
        index="{data}/scip/{run}/indices/index.npy",
        hpo_grid="{data}/scip/{run}/grid/rsh.pickle"
    output:
        "{data}/scip/{run}/models/xgb.pickle"
    notebook:
        "{data}/downstream_analysis/Stain-free Leukocyte Prediction.ipynb"