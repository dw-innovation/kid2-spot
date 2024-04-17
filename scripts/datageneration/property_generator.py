from typing import List

import numpy as np
from datageneration.data_model import TagAttributeExample, TagAttribute, Property


# numerical value generator
def get_random_decimal_with_metric(range):
    '''
    TODO: this should be reworked -- threshold should be defined based on metric
    '''
    h_ = np.random.choice(np.arange(range), 1)[0]
    if np.random.choice([True, False], 1)[0]:
        h_ = h_ / np.random.choice([10, 100], 1)[0]

    h_ = str(h_) + " " + np.random.choice(["mm", "cm", "m", "km", "in", "ft", "yd", "mi", "le"], 1)[0]  # "cm",
    return h_


def get_random_integer(max_value: int, min_value: int) -> int:
    return np.random.choice(np.arange(max_value), min_value)[0]


class PropertyGenerator:
    def __init__(self, named_property_examples: List[TagAttributeExample]):
        self.named_property_examples = named_property_examples

    def select_named_property_example(self, property_name: str) -> List[str]:
        # print(self.named_property_examples)
        for item in self.named_property_examples:
            if item['key'] == property_name:
                return item['examples']

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

        if "***any***" in tag_attribute.value: # examples only occur when the value is "any"
            if not attribute_examples:
                raise ValueError(f"There is an issue with {tag_attribute}")

            np.random.shuffle(attribute_examples)
            selected_example = attribute_examples[0]
            return Property(key=tag_attribute.key, operator=tag_attribute.operator, value=selected_example)
        else:
            return Property(key=tag_attribute.key, operator=tag_attribute.operator, value=tag_attribute.value)

    def generate_proper_noun_property(self, tag_attribute: TagAttribute) -> Property:
        '''Proper nouns are names such as name=Laughen_restaurant'''
        return self.generate_non_numerical_property(tag_attribute)

    def generate_numerical_property(self, tag_attribute: TagAttribute) -> Property:
        # todo --> we might need specific numerical function if we need to define logical max/min values.
        key_attribute = tag_attribute.key
        if "height" in key_attribute:
            # todo rename this
            generated_numerical_value = get_random_decimal_with_metric(2000)
        else:
            # todo rename this
            generated_numerical_value = str(get_random_integer(max_value=50, min_value=1))
        return Property(key=tag_attribute.key, operator=tag_attribute.operator, value=generated_numerical_value)

    def generate_color_property(self, tag_attribute: TagAttribute) -> Property:
        raise NotImplemented

    def run(self, tag_attribute: TagAttribute) -> Property:
        '''
        Generate a property based on a tag attribute.

        Parameters:
            tag_attribute (TagAttribute): The tag attribute object containing information about the attribute.

        Returns:
            Property: The generated property.

        This method checks the type of the tag attribute and generates a property accordingly.
        If the attribute is numeric, it generates a numerical property.
        If the attribute key contains 'name', it generates a proper noun property.
        If the attribute key contains 'color', it generates a color property.
        Otherwise, it generates a named property.
        '''
        if 'numeric' in tag_attribute.value:
            generated_property = self.generate_numerical_property(tag_attribute)
        else:
            if 'name' in tag_attribute.key:
                generated_property = self.generate_proper_noun_property(tag_attribute)
            elif 'color' in tag_attribute.key:
                generated_property = self.generate_color_property(tag_attribute)
            else:
                generated_property = self.generate_named_property(tag_attribute)
        return generated_property
