import json
import os
import unittest

from datageneration.generate_combination_table import QueryCombinationGenerator, write_output
from pytest import approx

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestGenerateCombination(unittest.TestCase):

    def setUp(self):
        geolocation_file_path = 'datageneration/tests/data/countries+states+cities.json'
        tag_list_path = 'datageneration/tests/data/Tag_List_v10.csv'
        arbitrary_value_list_path = 'datageneration/tests/data/Arbitrary_Value_List_v10.csv'
        self.query_comb_generator = QueryCombinationGenerator(geolocations_file_path=geolocation_file_path,
                                                              tag_list_path=tag_list_path,
                                                              arbitrary_value_list_path=arbitrary_value_list_path)

    def test_generate_random_tag_combinations(self):
        # todo: not a really test case, find an usecase
        num_queries = 100
        generated_tag_combs = self.query_comb_generator.generate_random_tag_combinations(num_queries)

        for comb in generated_tag_combs:
            print(comb)

    def test_write_to_file(self):
        test_samples = [{'a': {'t': 'area', 'v': 'Ceiba'},
                         'ns': [{'id': 0, 'flts': [{'k': 'landuse', 'v': 'grass', 'op': '=', 'n': 'lawn'}], 't': 'nwr'},
                                {'id': 1,
                                 'flts': [{'k': 'railway', 'v': 'railway_crossing', 'op': '=', 'n': 'level crossing'}],
                                 't': 'nwr'}, {'id': 2, 'flts': [
                                 {'k': 'social_facility:for', 'v': 'senior', 'op': '=', 'n': 'convalescent hospital'}],
                                               't': 'nwr'},
                                {'id': 3,
                                 'flts': [{'k': 'amenity', 'v': 'theatre', 'op': '=', 'n': 'performing arts center'}],
                                 't': 'nwr'}], 'es': [{'src': 0, 'tgt': 1, 't': 'dist', 'dist': '825 km'},
                                                      {'src': 1, 'tgt': 2, 't': 'dist', 'dist': '6.67 cm'},
                                                      {'src': 2, 'tgt': 3, 't': 'dist', 'dist': '2.98 yd'}]},
                        {'a': {'t': 'area', 'v': 'Plymouth'},
                         'ns': [
                             {'id': 0, 'flts': [{'k': 'water', 'v': 'river', 'op': '=', 'n': 'riverbank'}], 't': 'nwr'},
                             {'id': 1, 'flts': [{'k': 'office', 'v': 'ngo', 'op': '=', 'n': 'office of a ngo'}],
                              't': 'nwr'},
                             {'id': 2, 'flts': [{'k': 'building', 'v': 'monastery', 'op': '=', 'n': 'abbey'}],
                              't': 'nwr'}],
                         'es': [{'src': 0, 'tgt': 1, 't': 'dist', 'dist': '723 mm'},
                                {'src': 1, 'tgt': 2, 't': 'dist', 'dist': '787 le'}]}]

        write_output(test_samples, 'datageneration/tests/data/tmp_output.jsonl')

        tmp_file = 'datageneration/tests/data/tmp_output.jsonl'

        with open(tmp_file, 'r') as json_file:
            predicted_results = list(json_file)

        with open('datageneration/tests/data/test_output.jsonl', 'r') as json_file:
            expected_results = list(json_file)

        for predicted_result, expected_result in zip(predicted_results, expected_results):
            assert json.loads(predicted_result) == json.loads(expected_result)

        os.remove(tmp_file)

    def test_area_generate(self):
        trials = 100
        bbox_count = 0
        area_count = 0
        area_chance = 0.9
        for i in range(trials):
            result = self.query_comb_generator.generate_area(area_chance=area_chance)
            if len(result['val']) == 0:
                bbox_count += 1
            else:
                area_count += 1
        assert area_count / 100 == approx(area_chance, rel=0.5)

    def test_query_comb_drawn(self):

        drawn_idx = self.query_comb_generator.desriptors_to_idx['restaurant']
        must_word = 'cuisine'
        self.check_appearence(drawn_idx, self.query_comb_generator, must_word)

        drawn_idx = self.search_value_contain_word(self.query_comb_generator.desriptors_to_idx, 'connecting roads')
        must_word = 'lanes'
        self.check_appearence(drawn_idx, self.query_comb_generator, must_word)

    def search_value_contain_word(self, desriptors_to_idx, word):
        for key, value in desriptors_to_idx.items():
            if word in key:
                return value

    def check_appearence(self, drawn_idx, query_comb_generator, must_word):
        print(f"drawn idx {drawn_idx}")
        trials = [50, 100]
        # 0.2, 0.5 does not work
        for trial in trials:
            print(f"Number of trial: {trial}")
            for attempt in range(trial):
                chosen_comb = []

                while len(chosen_comb) <= trial:
                    comb = query_comb_generator.get_combs(drawn_idx=drawn_idx, max_number_combs=5)
                    if comb not in chosen_comb:
                        for comb_i in comb:
                            chosen_comb.extend([query_comb_generator.index_to_descriptors(i).strip() for i
                                                in comb_i.split('|')])

                must_word_presence = False
                for item in chosen_comb:

                    if must_word in item:
                        must_word_presence = True
                        continue

                assert must_word_presence


if __name__ == '__main__':
    unittest.main()
