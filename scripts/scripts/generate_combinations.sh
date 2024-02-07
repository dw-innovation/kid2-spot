python -m datageneration.generate_combination_table \
--geolocations_file_path datageneration/data/countries+states+cities.json \
--tag_list_path datageneration/data/Tag_List_v10.csv \
--arbitrary_value_list_path datageneration/data/Arbitrary_Value_List_v10.csv \
--output_file datageneration/results/IMR_Dataset_v12/samples.jsonl \
--area_chance 0.9 \
--write_output \
--max_number_tags_per_query 4 \
--samples 15000