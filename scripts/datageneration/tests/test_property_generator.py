import unittest
import pandas as pd
from datageneration.data_model import TagAttribute, Property
from datageneration.property_generator import PropertyGenerator

'''Run python -m unittest datageneration.tests.test_property_generator'''

class TestPropertyGenerator(unittest.TestCase):
    def setUp(self):
        attribute_examples_path = 'datageneration/tests/data/att_examples_test.jsonl'
        attribute_examples = pd.read_json(attribute_examples_path, lines=True).to_dict('records')
        self.property_generator = PropertyGenerator(named_property_examples=attribute_examples)

    def test_named_property(self):
        tag_attribute = TagAttribute(**{"key": "cuisine", "operator": "=", "value": "***any***"})
        named_property = self.property_generator.generate_named_property(tag_attribute)
        assert isinstance(named_property, Property)
