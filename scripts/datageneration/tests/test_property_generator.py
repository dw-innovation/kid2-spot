import unittest

import pandas as pd
from datageneration.data_model import TagAttribute, Property, Tag
from datageneration.property_generator import PropertyGenerator

'''Run python -m unittest datageneration.tests.test_property_generator'''


class TestPropertyGenerator(unittest.TestCase):
    def setUp(self):
        attribute_examples_path = 'datageneration/tests/data/att_examples_test.jsonl'
        attribute_examples = pd.read_json(attribute_examples_path, lines=True).to_dict('records')
        self.property_generator = PropertyGenerator(named_property_examples=attribute_examples)

    def test_named_property(self):
        tag_attribute = TagAttribute(**{"descriptors": ["cuisine"], "tags": [
            Tag(**{"key": "cuisine", "operator": "=", "value": "***any***"})]})
        named_property = self.property_generator.generate_named_property(tag_attribute)
        assert isinstance(named_property, Property)

    def test_proper_noun_property(self):
        tag_attribute = TagAttribute(descriptors=["name"], tags=[Tag(key="name", operator="~", value="***any***")])
        named_property = self.property_generator.generate_proper_noun_property(tag_attribute)
        assert isinstance(named_property, Property)
        assert named_property.value

    def test_numerical_property(self):
        numerical_tag_attribute = TagAttribute(**{"descriptors": ["height"], "tags": [
            Tag(**{"key": "height", "operator": "=", "value": "***numeric***"})]})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_attribute)

        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

        numerical_tag_attribute = TagAttribute(
            **{"descriptors": ["lanes in each direction", "lanes going in each direction"], "tags": [
                Tag(**{"key": "lanes:forward", "operator": "=", "value": "***numeric***"}),
                Tag(**{"key": "lanes:backward", "operator": "=", "value": "***numeric***"})
            ]})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_attribute)
        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

        # todo write asserting about type, min, max
        numerical_tag_attribute = TagAttribute(**{"descriptors": ["building levels", "floors"], "tags": [
            Tag(**{"key": "building:levels", "operator": "=", "value": "***numeric***"})]})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_attribute)
        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

    def test_other_type_property(self):
        # todo write an unit test, how the output would be
        other_type_tag_attribute = TagAttribute(**{
            "descriptors": ["traffic signals", "traffic control", "traffic lights", "traffic lamps", "signal lights",
                            "stop lights"], "tags": [
                Tag(**{"key": "highway", "operator": "=", "value": "traffic_signals"})]})
        other_property = self.property_generator.run(other_type_tag_attribute)
        assert isinstance(other_property, Property)

    def test_color_property(self):
        pass
