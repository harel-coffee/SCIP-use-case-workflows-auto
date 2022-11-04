# SCIP use case workflows

This repository contains all code to reproduce the use cases presented in
[A scalable, reproducible and open-source pipeline for morphologically profiling image cytometry data](https://www.biorxiv.org/content/10.1101/2022.10.24.512549v1).

It contains
- a package `scip_workflows` created using [nbdev](https://nbdev.fast.ai/),
- and three [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflows.

## Installation

To execute the workflows in this repository you need to [install Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

The workflows have been tested with Python 3.9.13.

## Usage

Reproducing the use cases is done by executing SCIP to profile the images, and Snakemake workflows to generate downstream analysis results.

The required configurations to run SCIP are in the [scip_configs](scip_configs) directory.

The following commands expect Snakemake to be available. Snakemake can be executed using conda environments or a pre-existing environment containing all required packages.

To reproduce a use-case, open a terminal where you cloned this repository and execute:
```bash
snakemake --configfile config/use_case.yaml --directory root_dir use_case
```
where
- `use_case` is one of `WBC`, `CD7` or `BBBC021`,
- `root_dir` points to where you downloaded the use case files

Make sure to update the config file to your situation; mainly setting the `parts` to the amount of output partitions SCIP generated.

This expects the environment to contain all required dependencies. Add `--use-conda` to let
Snakemake create a conda environment containing all requirements.

### Use case: WBC

Data and features (for SCIP and IDEAS) can be downloaded at the [Bioimage Archive](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD452)

### Use case: CD7

Data and SCIP features can be downloaded at the [Bioimage Archive](https://www.ebi.ac.uk/biostudies/studies/S-BIAD505)

### Use case: BBBC021

Data can be downloaded at the [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC021). Features can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7276510). You can download metadata BBBC021_v1_image.csv and BBBC021_v1_moa.csv from the supplementary materials "Data S2" in [[1]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3884769/).

[1] Ljosa, V., Caie, P. D., Ter Horst, R., Sokolnicki, K. L., Jenkins, E. L., Daya, S., ... & Carpenter, A. E. (2013). Comparison of methods for image-based profiling of cellular morphological responses to small-molecule treatment. Journal of biomolecular screening, 18(10), 1321-1329.

