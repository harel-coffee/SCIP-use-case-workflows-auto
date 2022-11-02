use rule preprocessing as WBC_preprocessing with:
    output: 
        "WBC_features.parquet"

rule WBC_labels:
    input:
        features="WBC_features.parquet",
        population_dir="../../meta"
    output:
        "WBC_labels.parquet"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/labels.ipynb"
    notebook:
        "../notebooks/WBC/labels.ipynb"

rule WBC_qc_paper_figure:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy"
    output:
        "figures/WBC_qc_masks.png"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/qc_paper_figure.ipynb"
    notebook:
        "../notebooks/WBC/qc_paper_figure.ipynb"

# rule WBC_feature_comparison:
#     input:
#         features="features.parquet",
#         columns="indices/columns.npy",
#         index="indices/index.npy",
#         ideas=""
#     conda:
#         "../envs/environment.yml"
#     threads: 1
#     log:
#         notebook="notebooks/feature_comparison.ipynb"
#     notebook:
#         "../notebooks/WBC/feature_comparison.ipynb"

rule WBC_all_hyperparameter_optimization:
    input:
        expand(
            "hpo/wbc_{grid}_{type}_{full}_{mask}_{model}.pickle",
            full=["full", "cyto"],
            grid=["rsh", "random"],
            mask=["otsu", "li", "otsuli"],
            type=["ideas", "scip"],
            model="xgboost"
        )

rule WBC_hyperparameter_optimization:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        labels="labels.parquet"
    output:
        "hpo/WBC_{grid}_{type}_{full}_{mask}_{model}.pickle"
    conda:
        "../envs/environment.yml"
    threads:
        10
    log:
        "hyperparameter_optimization_{grid}_{type}_{full}_{mask}_{model}.log"
    script:
        "../scripts/WBC/xgb_parameter_search.py"

rule WBC_classification:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        hpo_grid="grid/rsh.pickle"
    output:
        "models/WBC_xgb.pickle"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/scip_classification.ipynb"
    notebook:
        "../notebooks/WBC/scip_classification.ipynb"
