import unittest

import pandas as pd
from datageneration.generate_combination_table import QueryCombinationGenerator

'''
Execute it as follows: python -m unittest datageneration.tests.test_generate_combination_table
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
                                                              attribute_examples=attribute_examples,
                                                              max_distance=2000)

    def test_generate_entities(self):
        entities = self.query_comb_generator.generate_entities(number_of_entities_in_prompt=3,
                                                                      max_number_of_props_in_entity=0)

        assert len(entities) == 3

        for entity in entities:
            assert len(entity.properties) == 0

        entities = self.query_comb_generator.generate_entities(number_of_entities_in_prompt=3,
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

        properties = self.query_comb_generator.generate_properties(candidate_attributes=candidate_attributes,
                                                                   num_of_props=4)

        assert len(properties) == 4

        properties = self.query_comb_generator.generate_properties(candidate_attributes=candidate_attributes,
                                                                   num_of_props=3)

        assert len(properties) == 3


if __name__ == '__main__':
    unittest.main()
