rule preprocessing:
    input:
        expand(
            "features.{part}.parquet",
            part=range(int(config['parts'])),
            allow_missing=True
        )
    output:
        "features.parquet"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/processing_scip_features.ipynb"
    notebook:
        "../notebooks/{config[use_case]}/processing_scip_features.ipynb"

rule quality_control:
    input:
        "features.parquet"
    output:
        columns="indices/columns.npy",
        index="indices/index.npy"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/quality_control.ipynb"
    notebook:
        "../notebooks/{config[use_case]}/quality_control.ipynb"