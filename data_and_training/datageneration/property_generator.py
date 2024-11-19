import numpy as np
import pandas as pd
from typing import List, Dict

from datageneration.data_model import TagPropertyExample, TagProperty, Property, ColorBundle
from datageneration.utils import get_random_integer, get_random_decimal_with_metric


def fetch_color_bundle(property_examples: List[TagPropertyExample], bundle_path: str)->Dict[str,List[str]]:
    data = pd.read_csv(bundle_path)
    data = data.to_dict('records')
    color_examples = [item for item in property_examples if 'colour' in item['key']]

    color_bundles = []
    color_bundles_with_tags = {}

    for color_bundle in data:
        color_bundles.append(ColorBundle(descriptors = [x.strip() for x in color_bundle['Colour Descriptors'].split(',')],
                    color_values = [x.strip() for x in color_bundle['Colour Descriptors'].split(',')]))

    for color_example in color_examples:
        color_example_key = color_example['key']
        related_color_examples = color_example['examples']

        related_colors = []
        for related_color_example in related_color_examples:
            for color_bundle in color_bundles:
                if related_color_example in color_bundle.descriptors:
                    related_colors.extend(color_bundle.descriptors)

        related_colors = list(set(related_colors))
        color_bundles_with_tags[color_example_key] = related_colors

    return color_bundles_with_tags


class PropertyGenerator:
    def __init__(self, named_property_examples: List[TagPropertyExample],
        color_bundles: List[ColorBundle]
        ):
        self.named_property_examples = named_property_examples
        self.color_bundles = color_bundles

        self.tasks = []

    def select_named_property_example(self, property_name: str) -> List[str]:
        for item in self.named_property_examples:
            if item['key'] == property_name:
                return item['examples']

    def generate_non_numerical_property(self, tag_properties) -> Property:
        # todo: ipek -- i noticed that we haven't assign operator is equal initially the solution should uncomment the below line
        # operator = '='
        descriptor = np.random.choice(tag_properties.descriptors, 1)[0]

        if tag_properties.tags[0].value != "***example***":
            return Property(name=descriptor)

        # In case of bundle "name + brand", randomly select one of them
        selected_property = np.random.choice(tag_properties.tags)
        tag = selected_property.key + selected_property.operator + selected_property.value
        property_examples = self.select_named_property_example(tag)
        if not property_examples:
            return Property(name=descriptor)
            # return Property(key=tag_property.key, operator=tag_property.operator,value=tag_property.value, name=tag_property.value)

        if "~***example***" in tag:
            operator = "~"
        elif "=***example***" in tag:
            operator = "="
        else:
            print("Something does not seem to be right. Please check operator of property ", tag, "!")

        selected_example = np.random.choice(property_examples)

        return Property(name=descriptor, operator=operator, value=selected_example)

    def generate_numerical_property(self, tag_property: TagProperty) -> Property:
        # todo --> we might need specific numerical function if we need to define logical max/min values.
        descriptor = np.random.choice(tag_property.descriptors, 1)[0]
        # operator = "="
        operator = np.random.choice([">", "=", "<"])
        tag = tag_property.tags[0]
        if tag.key == "height":
            # todo rename this
            generated_numerical_value = get_random_decimal_with_metric(max_digits=5)
            generated_numerical_value = f'{generated_numerical_value.magnitude} {generated_numerical_value.metric}'
        else:
            # todo rename this
            generated_numerical_value = str(get_random_integer(max_digits=3))

        return Property(name=descriptor, operator=operator, value=generated_numerical_value)
        # return Property(key=tag_property.key, operator=tag_aproperty.operator, value=generated_numerical_value, name=tag_property.key)

    def generate_color_property(self, tag_attribute: TagProperty) -> Property:
        bundles_to_select = []
        for tag in tag_attribute.tags:
            tag_key = f'{tag.key}{tag.operator}{tag.value}'
            bundles_to_select.extend(self.color_bundles[tag_key])
        selected_color = np.random.choice(bundles_to_select, 1)[0]
        selected_descriptor = np.random.choice(tag_attribute.descriptors)
        return Property(name=selected_descriptor, operator='=', value=selected_color)

    def categorize_properties(self, tag_properties: List[TagProperty]):
        '''
        This function categorize the tag properties of an osm tag into three main group.
        :param tag_properties:
        :return:
        '''
        categories = {}
        for tag_property in tag_properties:
            tag_property_tags = tag_property.tags
            for tag_property_tag in tag_property_tags:
                if tag_property_tag.value == '***numeric***':
                    if 'numerical' not in categories:
                        categories['numerical'] = []
                    categories['numerical'].append(tag_property)
                elif 'brand' == tag_property_tag.key or 'name' == tag_property_tag.key or 'addr:housenumber' == tag_property_tag.key:
                    if 'popular_non_numerical' not in categories:
                        categories['popular_non_numerical'] = []
                    categories['popular_non_numerical'].append(tag_property)
                elif 'colour' in tag_property_tag.key:
                    if 'color' not in categories:
                        categories['colour'] = []
                    categories['colour'].append(tag_property)
                elif 'cuisine' in tag_property_tag.key:
                    if 'rare_non_numerical' not in categories:
                        categories['rare_non_numerical'] = []
                    categories['rare_non_numerical'].append(tag_property)
                else:
                    if 'other_non_numerical' not in categories:
                        categories['other_non_numerical'] = []
                    categories['other_non_numerical'].append(tag_property)
        return categories

    def run(self, tag_property: TagProperty) -> Property:
        '''
        Generate a property based on a tag property.

        Parameters:
            tag_property (TagProperty): The tag property object containing information about the property.

        Returns:
            Property: The generated property.

        This method checks the type of the tag property and generates a property accordingly.
        If the property is numeric, it generates a numerical property.
        If the property key contains 'name', it generates a proper noun property.
        If the property key contains 'color', it generates a color property.
        Otherwise, it generates a named property.
        '''
        # if '***numeric***' in tag_property.value:
        if any(t.value == '***numeric***' for t in tag_property.tags):
            generated_property = self.generate_numerical_property(tag_property)
        else:
            if any('colour' in t.key for t in tag_property.tags):
                generated_property = self.generate_color_property(tag_property)
            else:
                generated_property = self.generate_non_numerical_property(tag_property)
        return generated_property
