import unittest
from datageneration.enrich_geo_database import request_wd_id, get_dict_of_non_roman_alternatives

'''Run python -m unittest datageneration.tests.test_enrich_geo_database'''


class TestAreaGenerator(unittest.TestCase):
    def setUp(self):
        pass

    def test_request_wd_id(self):
        wd = request_wd_id('Zvishavane District')
        self.assertEqual(wd, 'Q24235929')


    def test_get_dict_of_non_roman_alternatives(self):
        wd = 'Q35997'
        results = get_dict_of_non_roman_alternatives(wd)
        assert 'ba' in results

        self.assertEqual(results['ba'], 'Измир')