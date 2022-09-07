# SCIP use case workflows

This repository contains Snakemake workflows to reproduce use cases presented in [insert paper].

It is built using two frameworks:
- nbdev
- Snakemake

nbdev makes is a framework for developing reusable code in notebooks. Functions are defined and
tested in notebooks, and exported to a package. This package can be installed and reused in other
notebooks or scripts.

Snakemake is a workflow framework to create reproducible data analyses. Workflows are defined
via a human-readable language, and can be easily executed in various environments.

## Installation



## Usage

The repository can be used to reproduce the use cases, or for development.

### Reproduce use cases

This involves executing the Snakemake workflows. The following expects snakemake to be available.
Snakemae can be executed using conda environments or a pre-existing environment
containing all required packages.

To reproduce a use-case, open a terminal in the current directory and execute:
```bash
snakemake --configfile config/use_case.yaml --directory root_dir
```
This expects the environment to contain all required dependencies. Add `--use-conda` to let
Snakemake create a conda environment containing all requirements.

### Development

This involves the nbdev framework.
