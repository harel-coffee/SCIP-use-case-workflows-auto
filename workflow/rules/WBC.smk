rule WBC_labels:
    input:
        features="features.parquet",
    output:
        "labels.parquet"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/labels.ipynb"
    notebook:
        "../notebooks/WBC/labels.ipynb"

rule WBC_qc_paper_figure:
    input:
        features="features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy"
    output:
        "figures/wbc_qc_masks.png"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/qc_paper_figure.ipynb"
    notebook:
        "../notebooks/WBC/qc_paper_figure.ipynb"

rule WBC_feature_comparison:
    input:
        features="features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        ideas=""
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/feature_comparison.ipynb"
    notebook:
        "notebooks/WBC/feature_comparison.ipynb"

# rule WBC_all_hyperparameter_optimization:
#     input:
#         expand(
#             "{data}/hpo/{grid}_{full}.pickle",
#             full=["full", "cyto"],
#             grid=["rsh", "random"],
#             data=config["data"]
#         )

# rule WBC_hyperparameter_optimization:
#     input:
#         features="{data_root}/{type}/{date}/features.parquet",
#         columns="{data_root}/{type}/{date}/indices/columns.npy",
#         index="{data_root}/{type}/{date}/indices/index.npy",
#         labels="{data_root}/{type}/{date}/labels.parquet"
#     output:
#         "{data_root}/{type}/{date}/hpo/{grid}_{full}.pickle"
#     conda:
#         "environment.yml"
#     params:
#         set=config["set"],
#         grid="{grid}"
#     threads:
#         10
#     log:
#         "{data_root}/{type}/{date}/log/hyperparameter_optimization_{grid}_{full}.log"
#     conda:
#         "../envs/environment.yml"
#     script:
#         "../scripts/python/{config[use_case]}/xgb_parameter_search.py"

rule WBC_classification:
    input:
        features="features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        hpo_grid="grid/rsh.pickle"
    output:
        "models/xgb.pickle"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/scip_classification.ipynb"
    notebook:
        "../notebooks/WBC/scip_classification.ipynb"