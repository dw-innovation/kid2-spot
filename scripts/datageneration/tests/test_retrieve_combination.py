import unittest

from datageneration.retrieve_combinations import CombinationRetriever

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestCombinationRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = CombinationRetriever(source='datageneration/tests/data/Primary_Keys_filtered9.xlsx',
                                              att_limit=100)

    def test_fetch_attributes(self):
        results = self.retriever.request_related_tag_attributes(tag_key='amenity', tag_value='restaurant', limit=100)

        processed_results = []
        for result in results:
            processed_results.append(f'{result.key}{result.operator}{result.value}')

        assert 'name=***any***' in processed_results
        assert 'cuisine=***any***' in processed_results
        assert 'building=water_tower' not in processed_results
        assert 'leisure=bowling_alley' not in processed_results

    def test_fetch_attributes_lanes(self):
        results = self.retriever.request_related_tag_attributes(tag_key='highway', tag_value='tertiary', limit=250)

        processed_results = []
        for result in results:
            processed_results.append(f'{result.key}{result.operator}{result.value}')

        assert 'name=***any***' in processed_results
        assert 'lanes=***numeric***' in processed_results
        assert 'bridge=yes' in processed_results
        assert 'cycleway=lane' in processed_results
        assert 'cycleway=separate' in processed_results
        assert 'cycleway:both=lane' in processed_results
        assert 'cycleway:both=opposite_lane' in processed_results
        assert 'sidewalk=right' in processed_results
        assert 'tunnel=yes' in processed_results
        assert 'lanes:psv>0' in processed_results
        assert 'building=water_tower' not in processed_results
        assert 'leisure=bowling_alley' not in processed_results

    def test_generate_attribute_examples(self):
        cuisine_examples = self.retriever.request_attribute_examples(attribute_key='cuisine', num_examples=50)
        assert len(cuisine_examples) == 50
        assert ';' not in cuisine_examples


if __name__ == '__main__':
    unittest.main()
