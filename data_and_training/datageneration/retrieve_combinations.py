import pandas as pd
import requests
import taginfo.query as ti
import unicodedata
from argparse import ArgumentParser
from diskcache import Cache
from tqdm import tqdm
from typing import List

from datageneration.data_model import Tag, TagProperty, TagCombination, TagPropertyExample, \
    remove_duplicate_tag_properties
from datageneration.utils import CompoundTagPropertyProcessor, SEPERATORS, write_output, split_descriptors

cache = Cache("tmp")

TAG_INFO_API_ENDPOINT = "https://taginfo.openstreetmap.org/api/4/tag/combinations?key=TAG_KEY&value=TAG_VALUE&sortname=together_count&sortorder=desc"


@cache.memoize()
def request_tag_combinations(tag_key, tag_value):
    '''
    Takes URL and sends request to website, returns the JSON response.

    :param str tag_key: Key value of the tag
    :param str tag_value: Key value of the value
    '''

    url = TAG_INFO_API_ENDPOINT.replace("TAG_KEY", tag_key).replace("TAG_VALUE", tag_value)

    response = requests.get(url)
    response.raise_for_status()

    if response.status_code == 200:
        return response.json()


def is_similar_to_english(char):
    '''
    Check if a given character only is part of the English alphabet or a slight variation thereof.

    :param char c: The char to be checked
    '''
    english_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    similar_chars = english_alphabet + 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'

    if char in similar_chars:
        return True

    normalized_char = unicodedata.normalize('NFKD', char)
    stripped_char = ''.join([c for c in normalized_char if not unicodedata.combining(c)])

    return stripped_char in similar_chars


def is_roman(s):
    '''
    Check if a given string only contains letters of the English alphabet or slight variations thereof.

    :param str s: The string to be checked
    '''
    for char in s:
        if ord(char) > 127 and not is_similar_to_english(char):
            return False
    return True


comp_prop_processor = CompoundTagPropertyProcessor()

def split_tags(tags: str) -> List[TagProperty]:
    '''this function splits the compound tags. it uses comp_prop_process for handling complex compounds such as highway'''
    processed_tags = set()
    for tag in tags.split(','):
        tag = tag.lstrip().strip()
        if 'AND' in tag:
            _tags = tag.split('AND')
            for _tag in _tags:
                _tag = _tag.lstrip().strip().replace(' ', '').lower()
                processed_tags.add(_tag)
        else:
            tag = tag.replace(' ', '').lower()
            if len(tag) == 0:
                continue

            if '[' in tag:
                compound_tag = comp_prop_processor.run(tag)
                for alt_tag in compound_tag:
                    processed_tags.add(alt_tag)
            else:
                processed_tags.add(tag)
    return list(processed_tags)


class CombinationRetriever(object):
    def __init__(self, source: str, prop_limit: int, min_together_count: int, add_non_roman_examples: bool):
        if source.endswith('xlsx'):
            tag_df = pd.read_excel(source, engine='openpyxl')
        else:
            tag_df = pd.read_csv(source, index_col=False)

        tag_df.drop_duplicates(subset='descriptors', inplace=True)
        tag_df["index"] = [i for i in range(len(tag_df))]
        all_osm_tags_and_properties = self.process_tag_properties(tag_df)

        self.tag_properties = self.fetch_tag_properties(tag_df)
        self.prop_limit = prop_limit
        self.min_together_count = min_together_count
        self.tag_df = tag_df
        self.all_osm_tags_and_properties = all_osm_tags_and_properties

        self.all_tags_property_ids = self.all_osm_tags_and_properties.keys()
        self.numeric_tags_property_ids = [f.split(">")[0] for f in filter(lambda x: x.endswith(">0"),
                                                                          self.all_tags_property_ids)]
        self.tags_requiring_many_examples = ["name~***example***", "brand~***example***", "addr:street~***example***",
                                             "addr:housenumber=***example***"]


        self.add_non_roman_examples = add_non_roman_examples

    def fetch_tag_properties(self, tag_df: pd.DataFrame) -> List[TagProperty]:
        '''
        Process properties from the Primary Key table (DataFrame)

        Args:
            tag_df (DataFrame): DataFrame containing tags and tag properties

        Returns:
            list: List of TagProperty objects
        '''
        tag_property_df = tag_df[tag_df['core/prop'] != 'core']
        tag_properties = []
        for tag_prop in tag_property_df.to_dict(orient='records'):
            descriptors = split_descriptors(tag_prop['descriptors'])
            splited_tags = split_tags(tag_prop['tags'])
            processed_tags = []

            for _tag in splited_tags:
                _tag_splits = None
                tag_operator = None

                for seperator in SEPERATORS:
                    if seperator in _tag:
                        _tag_splits = _tag.split(seperator)
                        tag_operator = seperator
                        continue
                processed_tags.append(Tag(key=_tag_splits[0], value=_tag_splits[1], operator=tag_operator))

            tag_properties.append(TagProperty(descriptors=descriptors, tags=processed_tags))
        return tag_properties

    def process_tag_properties(self, tag_df):
        """
        Process tags, properties from a DataFrame (PrimaryKey table).

        Args:
            tag_df (DataFrame): DataFrame containing tag properties.

        Returns:
            dict: Dictionary containing processed tag properties.
        """
        # all tags and properties
        all_osm_tags_and_properties = {}
        for tags in tag_df.to_dict(orient='records'):
            tag_type = tags['core/prop']
            if isinstance(tag_type, float):
                print(f'{tags} has no type, might be an invalid')
                continue
            tags_list = tags['tags']
            descriptors = tags['descriptors']
            tag_type = tag_type.strip()
            splited_tags = split_tags(tags['tags'])
            for _tag in splited_tags:
                _tag_splits = None
                tag_operator = None
                for seperator in SEPERATORS:
                    if seperator in _tag:
                        _tag_splits = _tag.split(seperator)
                        tag_operator = seperator
                        continue

                if _tag in all_osm_tags_and_properties:
                    if all_osm_tags_and_properties[_tag]["core/prop"] != tag_type:
                        all_osm_tags_and_properties[_tag] = {'tags': tags_list, 'key': _tag_splits[0],
                                                             'operator': tag_operator,
                                                             'value': _tag_splits[1],
                                                             'core/prop': "core/prop", 'descriptors': descriptors}
                else:
                    all_osm_tags_and_properties[_tag] = {'tags': tags_list, 'key': _tag_splits[0],
                                                         'operator': tag_operator, 'value': _tag_splits[1],
                                                         'core/prop': tag_type, 'descriptors': descriptors}

        return all_osm_tags_and_properties

    def request_property_examples(self, property_key: str, num_examples: int, count_limit: int = -1) -> List[str]:
        """
        It is a helper function for generate_property_examples. Retrieve examples of property keys. For example: cuisine -> italian, turkish, etc.

        Args:
            property_key (str): The key of the property for which examples are requested.
            num_examples (int): The number of examples to retrieve.

        Returns:
            List[str]: A list of property examples.

        This method fetches examples associated with the non-numerical properties from TagInfo API. It retrieves examples recursively page by page until the number of examples are equal to the threshold. The examples
        are split by semicolons (';'), and only examples that pass the 'isRoman' function
        check are included.
        """

        def fetch_examples_recursively(curr_page, fetched_examples):
            examples = ti.get_page_of_key_values(property_key, curr_page)
            if len(examples) == 0:
                return fetched_examples
            for example in examples:
                example_value = example['value']
                if count_limit !=-1:
                    example_count = example['count']
                    if example_count < count_limit:
                        continue

                for _example in example_value.split(';'):
                    if len(fetched_examples) > num_examples - 1:
                        return fetched_examples

                    if not self.add_non_roman_examples:
                        if is_roman(_example):
                            fetched_examples.add(_example)
                    else:
                        fetched_examples.add(_example)
            # Fetch next page recursively
            return fetch_examples_recursively(curr_page + 1, fetched_examples)

        fetched_examples = set()
        fetched_examples = fetch_examples_recursively(1, fetched_examples)
        return list(fetched_examples)

    def generate_property_examples(self, num_examples: int = 100000) -> List[TagPropertyExample]:
        """
        Generate property examples for each tags whose type is 'prop' or 'core/prop'.

        Args:
            num_examples (int): Number of examples to generate for each property (default is 100).

        Returns:
            List[TagPropertyExample]: List of TagPropertyExample objects containing property keys and their examples.

        This method generates examples for specific tag properties based on predefined criteria.
        It iterates through all tag properties and retrieves examples using the `request_property_examples`
        method. Examples are only generated for tag properties with type other than 'core' and having the value
        'numerical'. TagPropertyExample objects are created for each property along with their examples,
        which are then returned as a list.

        """
        properties_and_their_examples = []
        for curr_tag, all_tags in self.all_osm_tags_and_properties.items():
            if curr_tag not in self.tags_requiring_many_examples:
                curr_num_examples = 100
            else:
                curr_num_examples = num_examples

            if all_tags['core/prop'] != 'core' and '***example***' in curr_tag:
                if all_tags['key'] in ['roof:colour', 'building:colour', 'colour']:
                    examples = self.request_property_examples(all_tags['key'], num_examples=curr_num_examples, count_limit=10000)

                else:
                    examples = self.request_property_examples(all_tags['key'], num_examples=curr_num_examples)
                properties_and_their_examples.append(
                    TagPropertyExample(key=curr_tag, examples=examples))
        return properties_and_their_examples

    def check_other_tag_in_properties(self, other_tag: str) -> tuple:
        '''
        check if the combination in the property list
        Args:
            other_tag (str): e.g. name=, name~
        Returns:
            tuple(bool, int): True and its index in self.tag_properties, False and its index -1 otherwise.
        '''
        exists = False
        results = []
        for tag_prop_idx, tag_prop in enumerate(self.tag_properties):
            for tag_prop_tag in tag_prop.tags:
                tag_prop_tag_value = tag_prop_tag.value
                if tag_prop_tag_value in ['***any***', '***example***', 'yes', '***numeric***']:
                    tag_prop_tag_value = ''
                if f'{tag_prop_tag.key}{tag_prop_tag.operator}{tag_prop_tag_value}' == other_tag:
                    exists = True
                    results.append(tag_prop_idx)
                    # return (exists, tag_prop_idx)
                elif f'{tag_prop_tag.key}{tag_prop_tag.operator}' == other_tag:
                    exists = True
                    results.append(tag_prop_idx)
                    # return (exists, tag_prop_idx)
        if exists:
            return (exists, results)
        else:
            return (exists, -1)

    def request_related_tag_properties(self, tag_key: str, tag_value: str, limit: int = 100) -> List[TagProperty]:
        combinations = request_tag_combinations(tag_key=tag_key, tag_value=tag_value)['data']
        selected_properties = []
        for combination in combinations:
            if len(selected_properties) == limit or combination["together_count"] < self.min_together_count:
                return list(selected_properties)

            for seperator in SEPERATORS:
                exist_property, prop_indices = self.check_other_tag_in_properties(
                    other_tag=combination['other_key'] + seperator + combination['other_value'])
                if exist_property:
                    break

            if exist_property:
                for prop_index in prop_indices:
                    fetched_tag_prop = self.tag_properties[prop_index]
                    selected_properties.append(fetched_tag_prop)
            # else:
            #     print(f'{combination} does not exist')
            #     if (combination['other_key'] in self.numeric_tags_properties_ids and
            #             combination['other_value'].isnumeric()):
            #         if int(combination['other_value']) > 0:
            #             rewritten_tag = combination['other_key'] + ">0"
            #
            #             print("rewritten tag")
            #             print(rewritten_tag)
        return selected_properties

    def generate_tag_list_with_properties(self) -> List[TagCombination]:
        """
        Generates a list of TagCombination objects with associated properties. Given core osm tag, it fetches the
        associated combinations. Next, the combinations with a type of "core" are discarded.

        Returns:
            List[TagCombination]: A list of TagCombination objects containing cluster ID, descriptors,
                                  combination type, tags, and tag properties.
        """
        tag_combinations = []

        for row in tqdm(self.tag_df.to_dict(orient='records'), total=len(self.tag_df)):
            cluster_id = row['index']
            is_area = True if row['area/point'] == 'area' else False
            descriptors = split_descriptors(row['descriptors'])
            comb_type = row['core/prop'].strip()
            tags = split_tags(row['tags'])

            processed_tags = []
            processed_properties = []
            for tag in tags:
                for sep in SEPERATORS:
                    if sep in tag:
                        tag_key, tag_value = tag.split(sep)
                        processed_tags.append(Tag(key=tag_key, operator=sep, value=tag_value))

                        if comb_type != 'prop':
                            tag_properties = self.request_related_tag_properties(tag_key=tag_key,
                                                                                 tag_value=tag_value,
                                                                                 limit=self.prop_limit)
                            processed_properties.extend(tag_properties)

            processed_properties = remove_duplicate_tag_properties(processed_properties)
            tag_combinations.append(
                TagCombination(cluster_id=cluster_id, is_area=is_area, descriptors=descriptors, comb_type=comb_type,
                               tags=processed_tags, tag_properties=processed_properties))
        return tag_combinations


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''

    parser = ArgumentParser()
    parser.add_argument('--source', help='domain-specific primary keys', required=True)
    parser.add_argument('--output_file', help='Path to save the tag list', required=True)
    parser.add_argument('--prop_limit', help='Enter the number of related tags to be fetched by taginfo', default=100)
    parser.add_argument('--min_together_count', help='The min together count for a combination to be considered',
                        default=5000, type=int)
    parser.add_argument('--prop_example_limit', help='Enter the number of example values of the properties',
                        default=100000, type=int)
    parser.add_argument('--generate_tag_list_with_properties', help='Generate tag list with properties',
                        action='store_true')
    parser.add_argument('--generate_property_examples', help='Generate property examples',
                        action='store_true')
    parser.add_argument('--add_non_roman_examples', action='store_true', default=True)

    args = parser.parse_args()

    source = args.source
    prop_limit = int(args.prop_limit)
    min_together_count = args.min_together_count
    prop_example_limit = args.prop_example_limit
    add_non_roman_examples = args.add_non_roman_examples
    output_file = args.output_file
    generate_tag_list_with_properties = args.generate_tag_list_with_properties
    generate_property_examples = args.generate_property_examples

    comb_retriever = CombinationRetriever(source=source, prop_limit=prop_limit, min_together_count=min_together_count, add_non_roman_examples=add_non_roman_examples)

    if generate_tag_list_with_properties:
        tag_combinations = comb_retriever.generate_tag_list_with_properties()
        write_output(generated_combs=tag_combinations, output_file=output_file)

    if generate_property_examples:
        prop_examples = comb_retriever.generate_property_examples(num_examples=prop_example_limit)
        write_output(generated_combs=prop_examples, output_file=output_file)
