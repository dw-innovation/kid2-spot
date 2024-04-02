from typing import List

import numpy as np
from datageneration.data_model import TagAttributeExample, TagAttribute, Property


class PropertyGenerator:
    def __init__(self, named_property_examples: List[TagAttributeExample]):
        self.named_property_examples = named_property_examples

    def select_named_property_example(self, property_name: str) -> List[str]:
        for item in self.named_property_examples:
            if item['key'] == property_name:
                return item['examples']
        return None

    def generate_named_property(self, tag_attribute: TagAttribute) -> Property:
        """
        Generate a Property object based on the given TagAttribute.

        This function selects a random example for the specified tag attribute
        combination, shuffles the examples to ensure randomness, and then
        constructs a Property object using the selected example.

        Parameters:
        - tag_attribute (TagAttribute): The TagAttribute object containing key, operator,
          and value information to generate the property.

        Returns:
        - Property: A Property object constructed from the selected example.

        Example:
        ```python
        tag_attr = TagAttribute(key='cuisine', operator='=', value='italian')
        property_obj = generate_named_property(tag_attr)
        print(property_obj)
        ```
        """
        return self.generate_non_numerical_property(tag_attribute)

    def generate_non_numerical_property(self, tag_attribute):
        attribute_examples = self.select_named_property_example(
            f'{tag_attribute.key}{tag_attribute.operator}{tag_attribute.value}')
        np.random.shuffle(attribute_examples)
        selected_example = attribute_examples[0]
        return Property(key=tag_attribute.key, operator=tag_attribute.operator, value=selected_example)

    def generate_proper_noun_property(self, tag_attribute: TagAttribute) -> Property:
        return self.generate_non_numerical_property(tag_attribute)

    def generate_numerical_property(self, tag_attribute: TagAttribute) -> Property:
        raise NotImplemented

    def generate_color_property(self, tag_attribute: TagAttribute) -> Property:
        raise NotImplemented
