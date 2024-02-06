python -m datageneration.generate_combination_table \
--geolocations_file_path datageneration/data/countries+states+cities.json \
--tag_list_path datageneration/data/Tag_List_v10.csv \
--arbitrary_value_list_path datageneration/data/Arbitrary_Value_List_v10.csv \
--output_folder datageneration/results/IMR_Dataset_v12 \
--area_chance 0.9 \
--write_output \
--training_samples 5000 \
--development_samples 100 \
--testing_samples 1000