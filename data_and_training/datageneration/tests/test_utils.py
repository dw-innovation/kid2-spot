import unittest

from datageneration.utils import CompoundTagPropertyProcessor, clean_up_query

'''
Execute it as follows: python -m datageneration.tests.test_utils
'''


class TestUtils(unittest.TestCase):
    def test_compound_tag_property(self):
        compound_tags = '''["sidewalk"|"sidewalk:right"|"sidewalk:left"|"sidewalk:both"|"sidewalk:foot"|"sidewalk:right:foot"|"sidewalk:left:foot"|"sidewalk:both:foot"]=["both"|"left"|"right"|"separate"|"yes"|"designated"]'''
        processor = CompoundTagPropertyProcessor()
        results = processor.run(compound_tags)

        for result in results:
            if '[' in result:
                raise ValueError('Result must not a list')

    def test_clean_up(self):
        input_yaml = {'area': {'type': 'area', 'value': 'Hickory Withe'},
                      'entities': [
                          {'id': 0, 'is_area': False, 'name': 'traffic count', 'type': 'nwr', 'properties': []},
                          {'id': 1, 'is_area': False, 'name': 'ski jump', 'type': 'nwr', 'properties': []}],
                      'relations': {
                          'relations': [
                              {'type': 'distance', 'source': 0, 'target': 1,
                               'value': {'magnitude': '4691', 'metric': 'in'}}],
                          'type': 'within_radius'}}

        expected_yaml = {
            "area": {
                "type": "area",
                "value": "Hickory Withe"
            },
            "entities": [
                {
                    "id": 0,
                    "name": "traffic count",
                    "type": "nwr"
                },
                {
                    "id": 1,
                    "name": "ski jump",
                    "type": "nwr"
                }
            ],
            "relations": [
                {
                    "type": "distance",
                    "source": 0,
                    "target": 1,
                    "value": "4691 in"
                }
            ]
        }

        generated_yaml = clean_up_query(input_yaml)
        self.assertEqual(expected_yaml, generated_yaml)


if __name__ == '__main__':
    unittest.main()
