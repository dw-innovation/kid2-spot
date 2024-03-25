echo Generate Tag List With Attributes
python -m datageneration.retrieve_combinations \
--source datageneration/data/Primary_Keys_filtered8.xlsx \
--tag_list datageneration/data/Tag_List_v12.csv \
--arbitrary_value_list datageneration/data/Arbitrary_Value_List_v12.csv \
--generate_tag_list_with_attributes