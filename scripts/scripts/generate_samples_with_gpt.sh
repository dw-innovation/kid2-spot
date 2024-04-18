for idx in $(seq 1 1);
do
python -m datageneration.gpt_data_generator \
--tag_list_path datageneration/data/Tag_List_v12.csv \
--arbitrary_value_list_path datageneration/data/Arbitrary_Value_List_v12.csv \
--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv \
--tag_query_file datageneration/results/v12/samples_chunk_${idx}.jsonl \
--output_file datageneration/results/v12/gpt_generations_chunk_${idx}.jsonl \
--persona_path datageneration/prompts/personas.txt \
--styles_path datageneration/prompts/styles.txt \
--prob_usage_of_relative_spatial_terms 0.4
done