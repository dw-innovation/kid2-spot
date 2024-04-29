import unittest

from datageneration.data_model import TagAttribute, Tag
from datageneration.retrieve_combinations import CombinationRetriever

'''
Execute it as follows: python -m unittest datageneration.tests.test_retrieve_combination
'''


class TestCombinationRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = CombinationRetriever(source='datageneration/tests/data/Primary_Keys_test.xlsx',
                                              att_limit=100)

    def compare_tags(self, tested_tags, all_tags):
        exists = False
        for all_tag in all_tags:
            if set(tested_tags) == set(all_tag.tags):
                exists = True
                return exists
        return exists

    def test_fetch_attributes_from_primary_key_table(self):
        tag_attributes = self.retriever.fetch_tag_attributes(tag_df=self.retriever.tag_df)
        assert tag_attributes
        must_exist_tag_attributes = [
            TagAttribute(descriptors=['lanes going in each direction', 'lanes in each direction'],
                         tags=[Tag(key='lanes:forward', operator='=', value='***numeric***'),
                               Tag(key='lanes:backward', operator='=', value='***numeric***')]),
            TagAttribute(descriptors=['cuisine'], tags=[Tag(key='cuisine', operator='=', value='***any***')]),
            TagAttribute(descriptors=['car lanes', 'traffic lanes', 'street lanes'],
                         tags=[Tag(key='lanes', operator='=', value='***numeric***')])
        ]

        for must_exist_tag_attribute in must_exist_tag_attributes:
            assert self.compare_tags(must_exist_tag_attribute.tags, tag_attributes)

    def test_check_other_tag_in_attributes(self):
        exists, _ = self.retriever.check_other_tag_in_attributes(other_tag='name~')
        assert exists

        exists, _ = self.retriever.check_other_tag_in_attributes(other_tag='name=')
        assert not exists

        exists, _ = self.retriever.check_other_tag_in_attributes(other_tag='outdoor_seating=')
        assert exists

        exists, _ = self.retriever.check_other_tag_in_attributes(other_tag='cuisine=')
        assert exists

        exists, _ = self.retriever.check_other_tag_in_attributes(other_tag='lanes=')
        assert exists

    def test_related_tag_attributes(self):
        results = self.retriever.request_related_tag_attributes(tag_key='amenity', tag_value='restaurant', limit=100)

        processed_results = []
        for result in results:
            processed_results.extend(list(map(lambda x: f'{x.key}{x.operator}{x.value}', result.tags)))

        assert 'name~***any***' in processed_results
        assert 'cuisine=***any***' in processed_results
        assert 'building=water_tower' not in processed_results
        assert 'leisure=bowling_alley' not in processed_results

    def test_fetch_attributes_lanes(self):
        results = self.retriever.request_related_tag_attributes(tag_key='highway', tag_value='tertiary', limit=250)
        processed_results = []
        for result in results:
            processed_results.extend(list(map(lambda x: f'{x.key}{x.operator}{x.value}', result.tags)))

        assert 'name~***any***' in processed_results
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
    # #
    def test_generate_attribute_examples(self):
        cuisine_examples = self.retriever.request_attribute_examples(attribute_key='cuisine', num_examples=50)
        assert len(cuisine_examples) == 50
        assert ';' not in cuisine_examples


if __name__ == '__main__':
    unittest.main()
