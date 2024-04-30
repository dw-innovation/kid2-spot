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

    def generate_non_numerical_property(self, tag_attribute) -> Property:
        descriptor = np.random.choice(tag_attribute.descriptors, 1)[0]
        tag = tag_attribute.tags[0].key + tag_attribute.tags[0].operator + tag_attribute.tags[0].value
        attribute_examples = self.select_named_property_example(tag)
        if not attribute_examples:
            return Property(name=descriptor)
            # return Property(key=tag_attribute.key, operator=tag_attribute.operator,value=tag_attribute.value, name=tag_attribute.value)

        if "~***any***" in tag:
            operator = "~"
        elif "=***any***" in tag:
            operator = "="
        else:
            print("Something does not seem to be right. Please check operator of attribute ", tag, "!")

        np.random.shuffle(attribute_examples)
        selected_example = attribute_examples[0]

        return Property(name=descriptor, operator=operator, value=selected_example)

    def generate_proper_noun_property(self, tag_attribute: TagAttribute) -> Property:
        '''Proper nouns are names such as name=Laughen_restaurant'''
        return self.generate_non_numerical_property(tag_attribute)

    def generate_numerical_property(self, tag_attribute: TagAttribute) -> Property:
        # todo --> we might need specific numerical function if we need to define logical max/min values.
        descriptor = np.random.choice(tag_attribute.descriptors, 1)[0]
        # operator = "="
        operator = np.random.choice([">", "=", "<"])
        tag = tag_attribute.tags[0]
        if tag.key == "height":
            # todo rename this
            generated_numerical_value = get_random_decimal_with_metric(99999)
        else:
            # todo rename this
            generated_numerical_value = str(get_random_integer(max_value=200, min_value=1))

        return Property(name=descriptor, operator=operator, value=generated_numerical_value)
        # return Property(key=tag_attribute.key, operator=tag_attribute.operator, value=generated_numerical_value, name=tag_attribute.key)

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
        # if '***numeric***' in tag_attribute.value:
        if any(t.value == '***numeric***' for t in tag_attribute.tags):
            generated_property = self.generate_numerical_property(tag_attribute)
        else:
            if any(t.key == 'name' for t in tag_attribute.tags):
                generated_property = self.generate_proper_noun_property(tag_attribute)
            elif any(t.key == 'color' for t in tag_attribute.tags):
                generated_property = self.generate_color_property(tag_attribute)
            else:
                generated_property = self.generate_named_property(tag_attribute)
        return generated_property
