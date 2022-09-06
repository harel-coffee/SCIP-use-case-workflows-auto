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
        notebook="logs/notebooks/processing_scip_features.ipynb"
    notebook:
        "../notebooks/{config[use_case]}/processing_scip_features.ipynb"