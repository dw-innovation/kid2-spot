import unittest

from datageneration.data_model import TagProperty, Tag
from datageneration.retrieve_combinations import CombinationRetriever

'''
Execute it as follows: python -m unittest datageneration.tests.test_retrieve_combination
'''


class TestCombinationRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = CombinationRetriever(source='datageneration/tests/data/Primary_Keys_test10.xlsx',
                                              prop_limit=100,
                                              min_together_count=1000,
                                              add_non_roman_examples=False)

    def compare_tags(self, tested_tags, all_tags):
        exists = False
        for all_tag in all_tags:
            if set(tested_tags) == set(all_tag.tags):
                exists = True
                return exists
        return exists

    def test_fetch_properties_from_primary_key_table(self):
        tag_properties = self.retriever.fetch_tag_properties(tag_df=self.retriever.tag_df)
        assert tag_properties

        must_exist_tag_properties = [
            TagProperty(descriptors=['lanes going in each direction', 'lanes in each direction'],
                        tags=[Tag(key='lanes:forward', operator='=', value='***numeric***'),
                               Tag(key='lanes:backward', operator='=', value='***numeric***')]),
            TagProperty(descriptors=['cuisine'], tags=[Tag(key='cuisine', operator='=', value='***example***')]),
            TagProperty(descriptors=['car lanes', 'traffic lanes', 'street lanes'],
                        tags=[Tag(key='lanes', operator='=', value='***numeric***')])
        ]

        for must_exist_tag_property in must_exist_tag_properties:
            assert self.compare_tags(must_exist_tag_property.tags, tag_properties)

    def test_check_other_tag_in_properties(self):
        exists, _ = self.retriever.check_other_tag_in_properties(other_tag='name~')
        assert exists

        exists, _ = self.retriever.check_other_tag_in_properties(other_tag='name=')
        assert not exists

        exists, _ = self.retriever.check_other_tag_in_properties(other_tag='outdoor_seating=')
        assert exists

        exists, _ = self.retriever.check_other_tag_in_properties(other_tag='cuisine=')
        assert exists

        exists, _ = self.retriever.check_other_tag_in_properties(other_tag='lanes=')
        assert exists

    def test_related_tag_properties(self):
        results = self.retriever.request_related_tag_properties(tag_key='amenity', tag_value='restaurant', limit=100)

        processed_results = []
        for result in results:
            processed_results.extend(list(map(lambda x: f'{x.key}{x.operator}{x.value}', result.tags)))

        assert 'name~***example***' in processed_results
        assert 'cuisine=***example***' in processed_results
        assert 'building=water_tower' not in processed_results
        assert 'leisure=bowling_alley' not in processed_results

    def test_fetch_properties_lanes(self):
        results = self.retriever.request_related_tag_properties(tag_key='highway', tag_value='tertiary', limit=250)
        processed_results = []
        for result in results:
            processed_results.extend(list(map(lambda x: f'{x.key}{x.operator}{x.value}', result.tags)))

        assert 'name~***example***' in processed_results
        assert 'lanes=***numeric***' in processed_results
        assert 'bridge=yes' in processed_results
        assert 'cycleway=lane' in processed_results
        assert 'cycleway=separate' in processed_results
        assert 'cycleway:both=lane' in processed_results
        assert 'cycleway:both=opposite_lane' in processed_results
        assert 'sidewalk=right' not in processed_results
        assert 'tunnel=yes' in processed_results
        assert 'lanes:psv>0' in processed_results
        assert 'building=water_tower' not in processed_results
        assert 'leisure=bowling_alley' not in processed_results
    #
    def test_generate_properties_examples(self):
        cuisine_examples = self.retriever.request_property_examples(property_key='cuisine', num_examples=50)
        assert len(cuisine_examples) == 50
        assert ';' not in cuisine_examples

    def test_generate_properties_examples(self):
        color_examples = self.retriever.request_property_examples(property_key='colour', num_examples=50, count_limit=10000)
        assert len(color_examples) <= 50
        assert ';' not in color_examples


if __name__ == '__main__':
    unittest.main()
