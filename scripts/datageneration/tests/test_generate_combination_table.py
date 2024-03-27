import unittest

import pandas as pd
from datageneration.generate_combination_table import QueryCombinationGenerator

'''
Execute it as follows: python -m tests.test_feature_generator
'''


class TestGenerateCombination(unittest.TestCase):

    def setUp(self):
        geolocations_path = 'datageneration/tests/data/countries+states+cities.json'
        tag_combinations_path = 'datageneration/tests/data/tag_combinations_test.jsonl'
        attribute_examples_path = 'datageneration/tests/data/att_examples_test.jsonl'

        geolocations = pd.read_json(geolocations_path).to_dict('records')
        tag_combinations = pd.read_json(tag_combinations_path, lines=True).to_dict('records')
        attribute_examples = pd.read_json(attribute_examples_path, lines=True).to_dict('records')

        self.query_comb_generator = QueryCombinationGenerator(geolocations=geolocations,
                                                              tag_combinations=tag_combinations,
                                                              attribute_examples=attribute_examples)

    # def test_area_generate(self):
    #     trials = 100
    #     bbox_count = 0
    #     area_count = 0
    #     area_chance = 0.9
    #     for i in range(trials):
    #         result = self.query_comb_generator.generate_area(area_chance=area_chance)
    #         if len(result.value) == 0:
    #             bbox_count += 1
    #         else:
    #             area_count += 1
    #     assert area_count / 100 == approx(area_chance, rel=0.5)

    def test_generate_entities(self):
        entities = self.query_comb_generator.generate_prompt_entities(number_of_entities_in_prompt=3,
                                                                      max_number_of_props_in_entity=0)

        assert len(entities) == 3

        for entity in entities:
            assert len(entity.properties) == 0

        entities = self.query_comb_generator.generate_prompt_entities(number_of_entities_in_prompt=3,
                                                                      max_number_of_props_in_entity=4)
        assert len(entities) == 3
        for entity in entities:
            assert len(entity.properties) >= 1
            assert len(entity.properties) <= 4

    def test_property_generate(self):
        candidate_attributes = [{'key': 'name', 'operator': '=', 'value': '***any***'},
                                {'key': 'addr:street', 'operator': '=', 'value': '***any***'},
                                {'key': 'addr:housenumber', 'operator': '=', 'value': '***any***'},
                                {'key': 'height', 'operator': '=', 'value': '***numeric***'}]

        properties = self.query_comb_generator.generate_properties(candidate_attributes=candidate_attributes, num_of_props=4)
        assert len(properties) == 4

        properties = self.query_comb_generator.generate_properties(candidate_attributes=candidate_attributes, num_of_props=3)
        assert len(properties) == 3

    # def test_query_comb_drawn(self):
    #     drawn_idx = self.query_comb_generator.descriptor2_ids['restaurant']
    #     must_word = 'cuisine'
    #     self.check_appearence(drawn_idx, self.query_comb_generator, must_word)
    #
    #     drawn_idx = self.search_value_contain_word(self.query_comb_generator.descriptor2_ids, 'connecting roads')
    #     must_word = 'lanes'
    #     self.check_appearence(drawn_idx, self.query_comb_generator, must_word)
    #
    # def search_value_contain_word(self, desriptors_to_idx, word):
    #     for key, value in desriptors_to_idx.items():
    #         if word in key:
    #             return value

    # def check_appearence(self, drawn_idx, query_comb_generator, must_word):
    #     print(f"drawn idx {drawn_idx}")
    #     trials = [50, 100]
    #     # 0.2, 0.5 does not work
    #     for trial in trials:
    #         print(f"Number of trial: {trial}")
    #         for attempt in range(trial):
    #             chosen_comb = []
    #
    #             while len(chosen_comb) <= trial:
    #                 comb = query_comb_generator.get_combs(drawn_idx=drawn_idx, max_number_combs=5)
    #                 if comb not in chosen_comb:
    #                     for comb_i in comb:
    #                         chosen_comb.extend([query_comb_generator.index_to_descriptors(i).strip() for i
    #                                             in comb_i.split('|')])
    #
    #             must_word_presence = False
    #             for item in chosen_comb:
    #
    #                 if must_word in item:
    #                     must_word_presence = True
    #                     continue
    #
    #             assert must_word_presence


if __name__ == '__main__':
    unittest.main()
