import unittest

from datageneration.utils import CompoundTagAttributeProcessor

'''
Execute it as follows: python -m datageneration.tests.test_retrieve_combination
'''


class TestUtils(unittest.TestCase):
    def test_compound_tag_attribute(self):
        compound_tags = '''["sidewalk"|"sidewalk:right"|"sidewalk:left"|"sidewalk:both"|"sidewalk:foot"|"sidewalk:right:foot"|"sidewalk:left:foot"|"sidewalk:both:foot"]=["both"|"left"|"right"|"separate"|"yes"|"designated"]'''
        processor = CompoundTagAttributeProcessor()
        results = processor.run(compound_tags)

        for result in results:
            if '[' in result:
                raise ValueError('Result must not a list')



if __name__ == '__main__':
    unittest.main()
