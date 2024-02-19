mkdir -p datageneration/results/v11/train_test
python -m datageneration.construct_train_test \
--input_folder datageneration/results/v11/chunks_generated_sentences \
--output_folder datageneration/results/v11/train_test \
--dev_samples 500