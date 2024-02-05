import unittest

from datageneration.generate_combination_table import QueryCombinationGenerator

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestGenerateCombination(unittest.TestCase):

    def test_query_comb_generator(self):
        geolocation_file_path = 'datageneration/tests/data/countries+states+cities.json'
        tag_list_path = 'datageneration/tests/data/Tag_List_v10.csv'
        arbitrary_value_list_path = 'datageneration/tests/data/Arbitrary_Value_List_v10.csv'
        query_comb_generator = QueryCombinationGenerator(geolocations_file_path=geolocation_file_path,
                                                         tag_list_path=tag_list_path,
                                                         arbitrary_value_list_path=arbitrary_value_list_path)

        drawn_idx = query_comb_generator.desriptors_to_idx['restaurant']

        trials = [5, 10, 20]

        # 0.2, 0.5 does not work
        comb_chances = [1.0, 0.8]
        chosen_comb = []

        for trial in trials:
            for attempt in range(trial):
                for comb_chance in comb_chances:
                    print("comb chance", comb_chance)
                    for comb in query_comb_generator.get_combs(drawn_idx=drawn_idx, comb_chance=comb_chance,
                                                               max_number_combs=5):
                        chosen_comb.extend(
                            [query_comb_generator.index_to_descriptors(i).strip() for i in comb.split('|')])

                cuisine_presence = False
                if 'cuisine' in chosen_comb:
                    cuisine_presence = True

                assert cuisine_presence


if __name__ == '__main__':
    unittest.main()
