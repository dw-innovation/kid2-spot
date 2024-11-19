# Benchmarking scripts scripts
## Introduction

This folder contains all scripts related to the benchmarking of model results, as well as for the generation of
instructions for human annotators to generate more benchmarking data. 

### Executing the scripts

The core script to generate benchmarking results is "evaluate_results.py". It can be executed via the script
"data_and_training/scripts/run_benchmarking.sh". Please adjust the settings in the shell script before running the
benchmarking. A sample benchmarking dataset called "gold_annotations_05112024.xlsx" can be found in the data folder.

To generate instructions for annotators to generate more benchmarking data, please execute the script 
"generate_instructions.py" directly.
