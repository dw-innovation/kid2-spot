import unittest

import pandas as pd
from datageneration.data_model import TagProperty, Property, Tag
from datageneration.property_generator import PropertyGenerator, fetch_color_bundle

'''Run python -m unittest datageneration.tests.test_property_generator'''


class TestPropertyGenerator(unittest.TestCase):
    def setUp(self):
        property_examples_path = 'datageneration/tests/data/prop_examples_test.jsonl'
        property_examples = pd.read_json(property_examples_path, lines=True).to_dict('records')
        color_bundles = fetch_color_bundle(property_examples = property_examples, bundle_path = 'datageneration/tests/data/colour_bundles.csv')
        self.property_generator = PropertyGenerator(named_property_examples=property_examples, color_bundles=color_bundles)

    def test_named_property(self):
        tag_property = TagProperty(**{"descriptors": ["cuisine"], "tags": [
            Tag(**{"key": "cuisine", "operator": "=", "value": "***example***"})]})
        named_property = self.property_generator.generate_non_numerical_property(tag_property)
        assert isinstance(named_property, Property)

    def test_proper_noun_property(self):
        tag_property = TagProperty(descriptors=["name"], tags=[Tag(key="name", operator="~", value="***example***")])
        named_property = self.property_generator.generate_non_numerical_property(tag_property)
        assert isinstance(named_property, Property)
        assert named_property.value

    def test_numerical_property(self):
        numerical_tag_property = TagProperty(**{"descriptors": ["height"], "tags": [
            Tag(**{"key": "height", "operator": "=", "value": "***numeric***"})]})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_property)

        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

        numerical_tag_property = TagProperty(
            **{"descriptors": ["lanes in each direction", "lanes going in each direction"], "tags": [
                Tag(**{"key": "lanes:forward", "operator": "=", "value": "***numeric***"}),
                Tag(**{"key": "lanes:backward", "operator": "=", "value": "***numeric***"})
            ]})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_property)
        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

        # todo write asserting about type, min, max
        numerical_tag_property = TagProperty(**{"descriptors": ["building levels", "floors"], "tags": [
            Tag(**{"key": "building:levels", "operator": "=", "value": "***numeric***"})]})
        numerical_property = self.property_generator.generate_numerical_property(numerical_tag_property)
        # todo write asserting about type, min, max
        assert isinstance(numerical_property, Property)

    def test_other_type_property(self):
        # todo write an unit test, how the output would be
        other_type_tag_property = TagProperty(**{
            "descriptors": ["traffic signals", "traffic control", "traffic lights", "traffic lamps", "signal lights",
                            "stop lights"], "tags": [
                Tag(**{"key": "highway", "operator": "=", "value": "traffic_signals"})]})
        other_property = self.property_generator.run(other_type_tag_property)
        assert isinstance(other_property, Property)

    def test_color_property(self):
        colour_tag_property = TagProperty(descriptors = ['color', 'colour'],
        tags = [Tag(key='roof:colour', operator='=', value='***example***'),
                Tag(key='building:colour', operator='=', value='***example***'),
                Tag(key='colour', operator='=', value='***example***')]
        )

        color_property = self.property_generator.generate_color_property(colour_tag_property)
        print(color_property)