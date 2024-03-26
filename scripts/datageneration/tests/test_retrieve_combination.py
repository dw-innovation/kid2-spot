import unittest

from datageneration.retrieve_combinations import CombinationRetriever

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestCombinationRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = CombinationRetriever(source='datageneration/tests/data/Primary_Keys_test.xlsx',
                                              att_limit=100)

    # def test_assign_combination(self):
    #     arbitrary_tag_list = ['cuisine=', 'building:levels=', 'building:material=', 'height=', 'footway=', 'footway=',
    #                           'highway=', 'railway=', 'path=', 'crossing:island=', 'highway=', 'cycleway=', 'highway=',
    #                           'highway=', '=', 'ford=', 'highway=', 'cutting=', 'addr:street=', 'name=', 'toll=',
    #                           'lanes=', 'lanes:forward=', 'sport=', 'healthcare:speciality=', 'religion=',
    #                           'denomination=']
    #
    #     tag_df = self.retriever.tag_df
    #
    #     bundle_list = self.retriever.tag_df.loc[
    #         (tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'tags'].tolist()
    #
    #     # print(bundle_list)
    #
    #     tag_list = [tag.strip() for candidate in bundle_list for tag in candidate.split(",") if
    #                 not any(t in tag for t in ["*", "[", " AND "]) or any(
    #                     t in tag for t in ["***any***", "***numeric***"])]
    #
    #     # check if cuisine exists in the restaurant combinations
    #     tag_key = 'amenity'
    #     tag_value = 'restaurant'
    #
    #     results = self.retriever.assign_combinations(arbitrary_tag_list, tag_key, tag_list, tag_value)
    #
    #     cuisine_exists = False
    #
    #     for result in results.split('|'):
    #         if 'cuisine' in self.retriever.index_to_descriptors(int(result)):
    #             cuisine_exists = True
    #             continue
    #
    #     assert cuisine_exists
    #
    #     # check highway example
    #     tag_key = 'highway'
    #     tag_value = 'motorway'
    #
    #     results = self.retriever.assign_combinations(arbitrary_tag_list, tag_key, tag_list, tag_value)
    #
    #     lane_exists = False
    #
    #     for result in results.split('|'):
    #         if 'lane' in self.retriever.index_to_descriptors(int(result)):
    #             lane_exists = True
    #             continue
    #
    #     assert lane_exists

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
