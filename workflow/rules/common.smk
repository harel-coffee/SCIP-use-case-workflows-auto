rule preprocessing:
    input:
        expand(
            "features.{part}.parquet",
            part=range(int(config['parts']))
        )
    threads: 1
    params:
        usecase = lambda _, output: output[0].split("_")[0]
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/processing_scip_features.ipynb"
    notebook:
        "../notebooks/{params[usecase]}/processing_scip_features.ipynb"

rule quality_control:
    input:
        "{usecase}_features.parquet"
    output:
        columns="indices/{usecase}_columns.npy",
        index="indices/{usecase}_index.npy"
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/{usecase}_quality_control.ipynb"
    notebook:
        "../notebooks/{wildcards.usecase}/quality_control.ipynb"