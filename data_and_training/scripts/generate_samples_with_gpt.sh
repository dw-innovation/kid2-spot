VERSION=v17_2

#python -m datageneration.gpt_data_generator \
#--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv \
#--tag_query_file datageneration/results/${VERSION}/samples_case_non_roman_areas.jsonl \
#--output_prompt_generations datageneration/results/${VERSION}/prompt_generations_case_non_roman_areas.jsonl \
#--output_gpt_generations datageneration/results/${VERSION}/gpt_generations_case_non_roman_areas.jsonl \
#--persona_path datageneration/prompts/personas.txt \
#--styles_path datageneration/prompts/styles.txt \
#--prob_usage_of_relative_spatial_terms 0.7 \
#--prob_usage_of_written_numbers 0.25 \
#--prob_distance_writing_no_whitespace 0.4 \
#--prob_distance_writing_with_full_metric 0.5 \
#--prob_of_typos 0.5 \
#--max_dist_digits 5 \
#--save_yaml_csv \
#--generate_sentences
## Options: --generate_prompts , --generate_sentences
#
#
#python -m datageneration.gpt_data_generator \
#--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv \
#--tag_query_file datageneration/results/${VERSION}/samples_case_contain_rels.jsonl \
#--output_prompt_generations datageneration/results/${VERSION}/prompt_generations_case_contain_rels.jsonl \
#--output_gpt_generations datageneration/results/${VERSION}/gpt_generations_case_contain_rels.jsonl \
#--persona_path datageneration/prompts/personas.txt \
#--styles_path datageneration/prompts/styles.txt \
#--prob_usage_of_relative_spatial_terms 0.7 \
#--prob_usage_of_written_numbers 0.25 \
#--prob_distance_writing_no_whitespace 0.4 \
#--prob_distance_writing_with_full_metric 0.5 \
#--prob_of_typos 0.5 \
#--max_dist_digits 5 \
#--save_yaml_csv \
#--generate_sentences

#python -m datageneration.gpt_data_generator \
#--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv \
#--tag_query_file datageneration/results/${VERSION}/samples_case_props_v3.jsonl \
#--output_prompt_generations datageneration/results/${VERSION}/prompt_generations_case_props_v3.jsonl \
#--output_gpt_generations datageneration/results/${VERSION}/gpt_generations_case_props_v3.jsonl \
#--persona_path datageneration/prompts/personas.txt \
#--styles_path datageneration/prompts/styles.txt \
#--prob_usage_of_relative_spatial_terms 0.9 \
#--prob_usage_of_written_numbers 0.25 \
#--prob_distance_writing_no_whitespace 0.4 \
#--prob_distance_writing_with_full_metric 0.3 \
#--prob_of_typos 0.4 \
#--max_dist_digits 5 \
#--save_yaml_csv \
#--generate_prompts \
#--generate_sentences


python -m datageneration.gpt_data_generator \
--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv \
--tag_query_file datageneration/results/${VERSION}/dataset_${VERSION}_10k.jsonl \
--output_prompt_generations datageneration/results/${VERSION}/prompts_dataset_${VERSION}_10k.jsonl \
--output_gpt_generations datageneration/results/${VERSION}/gpt_generations_dataset_${VERSION}_10k.jsonl \
--persona_path datageneration/prompts/personas.txt \
--styles_path datageneration/prompts/styles.txt \
--prob_usage_of_relative_spatial_terms 0.4 \
--prob_usage_of_written_numbers 0.5 \
--prob_distance_writing_no_whitespace 0.4 \
--prob_distance_writing_with_full_metric 0.5 \
--prob_of_typos 0.4 \
--max_dist_digits 5 \
--save_yaml_csv \
--generate_prompts \
--generate_sentences
