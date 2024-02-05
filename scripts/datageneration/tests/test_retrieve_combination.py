import unittest

from datageneration.retrieve_combinations import CombinationRetriever

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestCombinationRetriever(unittest.TestCase):

    def test_assign_combination(self):
        retriever = CombinationRetriever(source='datageneration/tests/data/Primary_Keys_test.xlsx', num_examples=100)

        arbitrary_tag_list = ['cuisine=', 'building:levels=', 'building:material=', 'height=', 'footway=', 'footway=',
                              'highway=', 'railway=', 'path=', 'crossing:island=', 'highway=', 'cycleway=', 'highway=',
                              'highway=', '=', 'ford=', 'highway=', 'cutting=', 'addr:street=', 'name=', 'toll=',
                              'lanes=', 'lanes:forward=', 'sport=', 'healthcare:speciality=', 'religion=',
                              'denomination=']

        tag_df = retriever.tag_df

        bundle_list = retriever.tag_df.loc[
            (tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'tags'].tolist()

        # print(bundle_list)

        tag_list = [tag.strip() for candidate in bundle_list for tag in candidate.split(",") if
                    not any(t in tag for t in ["*", "[", " AND "]) or any(
                        t in tag for t in ["***any***", "***numeric***"])]


        # check if cuisine exists in the restaurant combinations
        tag_key = 'amenity'
        tag_value = 'restaurant'

        results = retriever.assign_combinations(arbitrary_tag_list, tag_key, tag_list, tag_value)

        cuisine_exists = False

        for result in results.split('|'):
            if 'cuisine' in retriever.index_to_descriptors(int(result)):
                cuisine_exists = True
                continue

        assert cuisine_exists

        # check highway example
        tag_key = ''
        tag_value = ''



if __name__ == '__main__':
    unittest.main()
