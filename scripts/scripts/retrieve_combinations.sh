echo Generate Tag List With Attributes
python -m datageneration.retrieve_combinations \
--source datageneration/data/Primary_Keys_filtered8.xlsx \
--output_file datageneration/data/tag_combinations_v12.jsonl \
--generate_tag_list_with_attributes