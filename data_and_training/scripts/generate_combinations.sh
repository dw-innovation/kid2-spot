VERSION=v17
SUFFIX=_2

## create non-roman samples with prob 1.0, increase prob_of_entities_with_props from 0.2 to 0.5
#python -m datageneration.generate_combination_table \
#--geolocations_file_path datageneration/data/countries+states+cities.json \
#--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
#--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
#--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
#--output_file datageneration/results/${VERSION}/samples_case_non_roman_areas.jsonl \
#--write_output \
#--max_distance_digits 5 \
#--max_number_of_entities_in_prompt 3 \
#--max_number_of_props_in_entity 3 \
#--prob_of_entities_with_props 0.5 \
#--prob_of_two_word_areas 0.5 \
#--prob_generating_contain_rel 0.5 \
#--ratio_within_radius_within 0.4 \
#--prob_adding_brand_names_as_entity 0.05 \
#--prob_of_numerical_properties 0.3 \
#--prob_of_color_properties 0.0 \
#--prob_of_non_numerical_properties 0.7 \
#--prob_of_non_roman_areas 1.0 \
#--samples 2000

##echo create entities with properties, reducing non-roman from 1.0 to 0.2
#python -m datageneration.generate_combination_table \
#--geolocations_file_path datageneration/data/countries+states+cities.json \
#--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
#--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
#--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
#--output_file datageneration/results/${VERSION}/samples_case_props_v2.jsonl \
#--write_output \
#--max_distance_digits 5 \
#--max_number_of_entities_in_prompt 3 \
#--max_number_of_props_in_entity 3 \
#--prob_of_entities_with_props 1.0 \
#--prob_of_two_word_areas 0.2 \
#--prob_generating_contain_rel 0.8 \
#--ratio_within_radius_within 0.4 \
#--prob_of_numerical_properties 0.1 \
#--prob_of_color_properties 0.0 \
#--prob_of_popular_non_numerical_properties 0.1 \
#--prob_of_other_non_numerical_properties 0.8 \
#--prob_adding_brand_names_as_entity 0.05 \
#--prob_of_non_roman_areas 0.2 \
#--samples 5000

##echo create entities with properties, reducing non-roman from 1.0 to 0.2
#python -m datageneration.generate_combination_table \
#--geolocations_file_path datageneration/data/countries+states+cities.json \
#--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
#--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
#--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
#--output_file datageneration/results/${VERSION}/samples_case_props_v3.jsonl \
#--write_output \
#--max_distance_digits 5 \
#--max_number_of_entities_in_prompt 4 \
#--max_number_of_props_in_entity 3 \
#--prob_of_entities_with_props 1.0 \
#--prob_of_two_word_areas 0.2 \
#--prob_generating_contain_rel 0.8 \
#--ratio_within_radius_within 0.4 \
#--prob_of_numerical_properties 0.1 \
#--prob_of_color_properties 0.0 \
#--prob_of_popular_non_numerical_properties 0.2 \
#--prob_of_other_non_numerical_properties 0.7 \
#--prob_adding_brand_names_as_entity 0.05 \
#--prob_of_non_roman_areas 0.2 \
#--samples 4000

## create entities with contain rels, reducing prop from 1.0 to 0.7
#python -m datageneration.generate_combination_table \
#--geolocations_file_path datageneration/data/countries+states+cities.json \
#--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
#--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
#--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
#--output_file datageneration/results/${VERSION}/samples_case_contain_rels.jsonl \
#--write_output \
#--max_distance_digits 5 \
#--max_number_of_entities_in_prompt 3 \
#--max_number_of_props_in_entity 3 \
#--prob_of_entities_with_props 0.7 \
#--prob_of_two_word_areas 0.5 \
#--prob_generating_contain_rel 1.0 \
#--ratio_within_radius_within 0.4 \
#--prob_of_numerical_properties 0.3 \
#--prob_of_color_properties 0.0 \
#--prob_of_non_numerical_properties 0.7 \
#--prob_adding_brand_names_as_entity 0.05 \
#--prob_of_non_roman_areas 0.2 \
#--samples 2000

## create color property
#python -m datageneration.generate_combination_table \
#--geolocations_file_path datageneration/data/countries+states+cities.json \
#--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
#--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
#--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
#--color_bundle_path datageneration/data/colour_bundles.csv \
#--output_file datageneration/results/${VERSION}/samples_case_contain_colors.jsonl \
#--write_output \
#--max_distance_digits 5 \
#--max_number_of_entities_in_prompt 3 \
#--max_number_of_props_in_entity 3 \
#--prob_of_entities_with_props 0.7 \
#--prob_of_two_word_areas 0.5 \
#--prob_generating_contain_rel 1.0 \
#--ratio_within_radius_within 0.4 \
#--prob_of_numerical_properties 0.0 \
#--prob_of_color_properties 1.0 \
#--prob_of_other_non_numerical_properties 0.0 \
#--prob_of_popular_non_numerical_properties 0.0 \
#--prob_adding_brand_names_as_entity 0.05 \
#--prob_of_non_roman_areas 0.2 \
#--samples 200

## create rare properties
#python -m datageneration.generate_combination_table \
#--geolocations_file_path datageneration/data/countries+states+cities.json \
#--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
#--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
#--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
#--color_bundle_path datageneration/data/colour_bundles.csv \
#--output_file datageneration/results/${VERSION}/samples_case_contain_cuisine.jsonl \
#--write_output \
#--max_distance_digits 5 \
#--max_number_of_entities_in_prompt 3 \
#--max_number_of_props_in_entity 3 \
#--prob_of_entities_with_props 0.7 \
#--prob_of_two_word_areas 0.5 \
#--prob_generating_contain_rel 1.0 \
#--ratio_within_radius_within 0.4 \
#--prob_of_numerical_properties 0.0 \
#--prob_of_color_properties 0.0 \
#--prob_of_rare_non_numerical_properties 1.0 \
#--prob_of_other_non_numerical_properties 0.0 \
#--prob_of_popular_non_numerical_properties 0.0 \
#--prob_adding_brand_names_as_entity 0.05 \
#--prob_of_non_roman_areas 0.2 \
#--samples 10


## create dataset (no cluster yet)
python -m datageneration.generate_combination_table \
--geolocations_file_path datageneration/data/countries+states+cities.json \
--non_roman_vocab_file_path datageneration/data/area_non_roman_vocab.json \
--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
--color_bundle_path datageneration/data/colour_bundles.csv \
--output_file datageneration/results/${VERSION}${SUFFIX}/dataset_${VERSION}${SUFFIX}_10k.jsonl \
--write_output \
--max_distance_digits 5 \
--max_number_of_entities_in_prompt 3 \
--max_number_of_props_in_entity 3 \
--prob_of_entities_with_props 0.6 \
--prob_of_two_word_areas 0.5 \
--prob_generating_contain_rel 0.4 \
--prob_of_numerical_properties 0.2 \
--prob_of_color_properties 0.2 \
--prob_of_rare_non_numerical_properties 0.2 \
--prob_of_other_non_numerical_properties 0.2 \
--prob_of_popular_non_numerical_properties 0.2 \
--prob_adding_brand_names_as_entity 0.025 \
--prob_of_non_roman_areas 0.3 \
--prob_of_cluster_entities 0.0 \
--samples 10000
