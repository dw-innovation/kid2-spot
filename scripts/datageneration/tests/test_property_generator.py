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

    def test_proper_noun_property(self):
        tag_attribute = TagAttribute(**{"key": "name", "operator": "=", "value": "***any***"})
        named_property = self.property_generator.generate_proper_noun_property(tag_attribute)
        assert isinstance(named_property, Property)

    def test_numerical_property(self):
        numerical_tag_attribute = TagAttribute(**{"key": "height", "operator": "=", "value": "***numeric***"})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_attribute)

        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

        numerical_tag_attribute = TagAttribute(**{"key": "lanes", "operator": "=", "value": "***numeric***"})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_attribute)
        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

        # todo write asserting about type, min, max
        numerical_tag_attribute = TagAttribute(**{"key": "building:levels", "operator": "=", "value": "***numeric***"})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_attribute)
        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

    def test_other_type_property(self):
        # todo write an unit test, how the output would be
        other_type_attribute = TagAttribute(**{"key": "highway", "operator": "=", "value": "passing_place"})
        self.property_generator.run(other_type_attribute)

    def test_color_property(self):
        pass
