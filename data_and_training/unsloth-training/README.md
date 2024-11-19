# Unsloth training script
## Introduction

Script to execute the fine-tuning of the Llama 3 (or other language) model with the artificially generated query data.

### Executing the script

These are the steps that need to be taken in order to execute the script.
1) Add the generated dataset files in the "data" folder. There should be two .tsv files for the train and the dev set.
2) Edit the training parameters and other settings in the shell script "scripts/train_unsloth.sh"
3) To allow for the pushing of the trained model to huggingface, create a ".env" file and add the token via a variable "HF_TOKEN" in the unsloth-training folder
4) While in the unsloth-training folder, execute the shell script via the command "sh scripts/train_unsloth.sh"
5) The final trained model will be saved in the folder "models", while test results are stored in "test_results"

The model is now trained and can be used for benchmarking or for deployment, e.g. via huggingface.