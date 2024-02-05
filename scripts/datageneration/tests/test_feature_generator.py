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
        must_word = 'cuisine'
        self.check_appearence(drawn_idx, query_comb_generator, must_word)

        drawn_idx = self.search_value_contain_word(query_comb_generator.desriptors_to_idx, 'connecting roads')
        must_word = 'lanes'
        self.check_appearence(drawn_idx, query_comb_generator, must_word)

    def search_value_contain_word(self, desriptors_to_idx, word):
        for key, value in desriptors_to_idx.items():
            if word in key:
                return value

    def check_appearence(self, drawn_idx, query_comb_generator, must_word):
        print(f"drawn idx {drawn_idx}")
        trials = [20, 50]
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
