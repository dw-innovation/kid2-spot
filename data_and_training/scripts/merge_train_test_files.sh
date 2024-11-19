#python -m datageneration.merge_train_test_files \
#--input_files datageneration/results/v15/train_fixed.tsv,datageneration/results/v15/dev_fixed.tsv,datageneration/results/v16_1/dev_props.tsv,datageneration/results/v16_1/train_props.tsv,datageneration/results/v16_2/train_props_v2.tsv,datageneration/results/v16_2/dev_props_v2.tsv \
#--output_folder datageneration/results/v16_2

#python -m datageneration.merge_train_test_files \
#--input_files datageneration/results/v15/train_fixed.tsv,datageneration/results/v15/dev_fixed.tsv,datageneration/results/v16_3/dev_props_v2.tsv,datageneration/results/v16_3/train_props_v2.tsv \
#--output_folder datageneration/results/v16_3

VERSIONONE=v16_3
VERSIONTWO=v17_1-2
OUTPUTVERSION=v16_3-17_1-2

python -m datageneration.merge_train_test_files \
--input_files datageneration/results/${VERSIONONE}/train_test/train_${VERSIONONE}.tsv,datageneration/results/${VERSIONONE}/train_test/dev_${VERSIONONE}.tsv,datageneration/results/${VERSIONTWO}/train_test/train_${VERSIONTWO}.tsv,datageneration/results/${VERSIONTWO}/train_test/dev_${VERSIONTWO}.tsv \
--output_folder datageneration/results/${OUTPUTVERSION}