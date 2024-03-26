import unittest

from datageneration.retrieve_combinations import CombinationRetriever

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestCombinationRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = CombinationRetriever(source='datageneration/tests/data/Primary_Keys_test.xlsx',
                                              att_limit=100)
    def test_fetch_attributes(self):
        results = self.retriever.request_related_tag_attributes(tag_key='amenity', tag_value='restaurant', limit=50)

        print("attributes are:")
        processed_results = []
        for result in results:
            processed_results.append(f'{result.key}{result.operator}{result.value}')

        assert 'name=***any***' in processed_results
        assert 'cuisine=***any***' in processed_results
        assert 'building=water_tower' not in processed_results
        assert 'leisure=bowling_alley' not in processed_results

if __name__ == '__main__':
    unittest.main()
