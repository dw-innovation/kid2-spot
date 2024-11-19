VERSION=v17_2

mkdir -p datageneration/results/${VERSION}/train_test
python -m datageneration.construct_train_test \
--input_file datageneration/results/${VERSION}/gpt_generations_dataset_${VERSION}_10k_yaml.jsonl \
--output_folder datageneration/results/${VERSION}/train_test \
--dev_samples 1000