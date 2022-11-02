# SCIP use case workflows

This repository contains Snakemake workflows to reproduce use cases presented in
[A scalable, reproducible and open-source pipeline for morphologically profiling image cytometry data](https://www.biorxiv.org/content/10.1101/2022.10.24.512549v1).

It is built using two frameworks:
- nbdev
- Snakemake

nbdev is a framework for developing reusable code in notebooks. Functions are defined and
tested in notebooks, and exported to a package. This package can be installed and reused in other
notebooks or scripts.

Snakemake is a workflow framework to create reproducible data analyses. Workflows are defined
via a human-readable language, and can be easily executed in various environments.

## Installation

To execute the workflows in this repository you need to [install Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)

## Usage

Reproducing the use cases is done by executing Snakemake workflows. The following expects Snakemake to be available. Snakemake can be executed using conda environments or a pre-existing environment
containing all required packages.

To reproduce a use-case, open a terminal where you cloned this repository and execute:
```bash
snakemake --configfile config/use_case.yaml --directory root_dir use_case
```
where `use_case` is one of `WBC`, `CD7` or `BBBC021`.

This expects the environment to contain all required dependencies. Add `--use-conda` to let
Snakemake create a conda environment containing all requirements.
