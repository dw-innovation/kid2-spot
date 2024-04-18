VERSION=v12

python -m datageneration.generate_combination_table \
--geolocations_file_path datageneration/data/countries+states+cities.json \
--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
--tag_attribute_examples_path datageneration/data/att_examples_${VERSION}.jsonl \
--output_file datageneration/results/${VERSION}/samples.jsonl \
--write_output \
--max_distance 2000 \
--max_number_of_props_in_entity 4 \
--number_of_entities_in_prompt 4 \
--samples 500