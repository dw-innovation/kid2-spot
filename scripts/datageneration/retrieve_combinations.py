import unicodedata
from argparse import ArgumentParser
from typing import List

import pandas as pd
import requests
import taginfo.query as ti
from datageneration.data_model import Tag, TagAttribute, TagCombination, TagAttributeExample, \
    remove_duplicate_tag_attributes
from datageneration.utils import CompoundTagAttributeProcessor, SEPERATORS, write_output
from diskcache import Cache
from tqdm import tqdm

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


def isSimilarToEnglish(char):
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


def isRoman(s):
    '''
    Check if a given string only contains letters of the English alphabet or slight variations thereof.

    :param str s: The string to be checked
    '''
    for char in s:
        if ord(char) > 127 and not isSimilarToEnglish(char):
            return False
    return True


comp_att_processor = CompoundTagAttributeProcessor()


def split_descriptors(descriptors: str) -> List[str]:
    '''this function splits the descriptors as a list of single descriptor'''
    processed_descriptors = set()

    for descriptor in descriptors.split('|'):
        descriptor = descriptor.lstrip().strip().lower()
        processed_descriptors.add(descriptor)

    return processed_descriptors


def split_tags(tags: str) -> List[TagAttribute]:
    '''this function splits the compound tags. it uses comp_attr_process for handling complex compounds such as highway'''
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
                compound_tag = comp_att_processor.run(tag)
                for alt_tag in compound_tag:
                    processed_tags.add(alt_tag)
            else:
                processed_tags.add(tag)
    return list(processed_tags)


class CombinationRetriever(object):
    def __init__(self, source, att_limit):
        if source.endswith('xlsx'):
            tag_df = pd.read_excel(source, engine='openpyxl')
        else:
            tag_df = pd.read_csv(source, index_col=False)

        tag_df.drop_duplicates(subset='descriptors', inplace=True)
        tag_df["index"] = [i for i in range(len(tag_df))]
        all_osm_tags_and_attributes = self.process_tag_attributes(tag_df)

        self.tag_attributes = self.fetch_tag_attributes(tag_df)
        self.att_limit = att_limit
        self.tag_df = tag_df
        self.all_osm_tags_and_attributes = all_osm_tags_and_attributes

        self.all_tags_attributes_ids = self.all_osm_tags_and_attributes.keys()
        self.numeric_tags_attributes_ids = [f.split(">")[0] for f in filter(lambda x: x.endswith(">0"),
                                                                            self.all_tags_attributes_ids)]

    def fetch_tag_attributes(self, tag_df: pd.DataFrame) -> List[TagAttribute]:
        '''
        Process attributes from the Primary Key table (DataFrame)

        Args:
            tag_df (DataFrame): DataFrame containing tags and tag attributes

        Returns:
            list: List of TagAttribute objects
        '''
        tag_attributes_df = tag_df[tag_df['type'] != 'core']
        tag_attributes = []
        for tag_attr in tag_attributes_df.to_dict(orient='records'):
            descriptors = split_descriptors(tag_attr['descriptors'])
            splited_tags = split_tags(tag_attr['tags'])
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

            tag_attributes.append(TagAttribute(descriptors=descriptors, tags=processed_tags))
        return tag_attributes

    def process_tag_attributes(self, tag_df):
        """
        Process tags, attributes from a DataFrame (PrimaryKey table).

        Args:
            tag_df (DataFrame): DataFrame containing tag attributes.

        Returns:
            dict: Dictionary containing processed tag attributes.
        """
        # all tags and attributes
        all_osm_tags_and_attributes = {}
        for tags in tag_df.to_dict(orient='records'):
            tag_type = tags['type']
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

                if _tag in all_osm_tags_and_attributes:
                    if all_osm_tags_and_attributes[_tag]["type"] != tag_type:
                        all_osm_tags_and_attributes[_tag] = {'tags': tags_list, 'key': _tag_splits[0],
                                                             'operator': tag_operator,
                                                             'value': _tag_splits[1],
                                                             'type': "core/attr", 'descriptors': descriptors}
                else:
                    all_osm_tags_and_attributes[_tag] = {'tags': tags_list, 'key': _tag_splits[0],
                                                         'operator': tag_operator, 'value': _tag_splits[1],
                                                         'type': tag_type, 'descriptors': descriptors}

        return all_osm_tags_and_attributes

    def request_attribute_examples(self, attribute_key: str, num_examples: int) -> List[str]:
        """
        It is a helper function for generate_attribute_examples. Retrieve examples of attribute keys. For example: cuisine -> italian, turkish, etc.

        Args:
            attribute_key (str): The key of the attribute for which examples are requested.
            num_examples (int): The number of examples to retrieve.

        Returns:
            List[str]: A list of attribute examples.

        This method fetches examples associated with the non-numerical attributes from TagInfo API. It retrieves examples recursively page by page until the number of examples are equal to the threshold. The examples
        are split by semicolons (';'), and only examples that pass the 'isRoman' function
        check are included.
        """

        def fetch_examples_recursively(curr_page, fetched_examples):
            examples = ti.get_page_of_key_values(attribute_key, curr_page)
            if len(examples) == 0:
                return fetched_examples
            for example in examples:
                example = example['value']
                for _example in example.split(';'):
                    if len(fetched_examples) > num_examples - 1:
                        return fetched_examples
                    if isRoman(_example):
                        fetched_examples.add(_example)
            # Fetch next page recursively
            return fetch_examples_recursively(curr_page + 1, fetched_examples)

        fetched_examples = set()
        fetched_examples = fetch_examples_recursively(1, fetched_examples)
        return list(fetched_examples)

    def generate_attribute_examples(self, num_examples: int = 100) -> List[TagAttributeExample]:
        """
        Generate attribute examples for each tags whose type is 'attr' or 'core/attr'.

        Args:
            num_examples (int): Number of examples to generate for each attribute (default is 100).

        Returns:
            List[TagAttributeExample]: List of TagAttributeExample objects containing attribute keys and their examples.

        This method generates examples for specific tag attributes based on predefined criteria.
        It iterates through all tag attributes and retrieves examples using the `request_attribute_examples`
        method. Examples are only generated for tag attributes with type other than 'core' and having the value
        'numerical'. TagAttributeExample objects are created for each attribute along with their examples,
        which are then returned as a list.

        """
        attributes_and_their_examples = []
        for key, value in self.all_osm_tags_and_attributes.items():
            if value['type'] != 'core' and '***any***' in key:
                examples = self.request_attribute_examples(value['key'], num_examples=num_examples)
                attributes_and_their_examples.append(
                    TagAttributeExample(key=key, examples=examples))
        return attributes_and_their_examples

    def check_other_tag_in_attributes(self, other_tag: str) -> tuple:
        '''
        check if the combination in the attribute list
        Args:
            other_tag (str): e.g. name=, name~
        Returns:
            tuple(bool, int): True and its index in self.tag_attributes, False and its index -1 otherwise.
        '''
        exists = False

        for tag_attr_idx, tag_attr in enumerate(self.tag_attributes):
            for tag_attr_tag in tag_attr.tags:
                tag_attr_tag_value = tag_attr_tag.value
                if tag_attr_tag.value in ['***any***', 'yes', '***numeric***']:
                    tag_attr_tag_value = ''
                if f'{tag_attr_tag.key}{tag_attr_tag.operator}{tag_attr_tag_value}' == other_tag:
                    exists = True
                    return (exists, tag_attr_idx)

        return (exists, -1)

    def request_related_tag_attributes(self, tag_key: str, tag_value: str, limit: str = 100) -> List[TagAttribute]:
        combinations = request_tag_combinations(tag_key=tag_key, tag_value=tag_value)['data']
        selected_attributes = []
        for combination in combinations:
            if len(selected_attributes) == limit:
                return list(selected_attributes)

            for seperator in SEPERATORS:
                exist_attribute, att_index = self.check_other_tag_in_attributes(
                    other_tag=combination['other_key'] + seperator + combination['other_value'])
                if exist_attribute:
                    break

            if exist_attribute:
                fetched_tag_attr = self.tag_attributes[att_index]
                selected_attributes.append(fetched_tag_attr)
            # else:
            #     print(f'{combination} does not exist')
            #     if (combination['other_key'] in self.numeric_tags_attributes_ids and
            #             combination['other_value'].isnumeric()):
            #         if int(combination['other_value']) > 0:
            #             rewritten_tag = combination['other_key'] + ">0"
            #
            #             print("rewritten tag")
            #             print(rewritten_tag)
        return selected_attributes

    def generate_tag_list_with_attributes(self) -> List[TagCombination]:
        """
        Generates a list of TagCombination objects with associated attributes. Given core osm tag, it fetches the
        associated combinations. Next, the combinations with a type of "core" are discarded.

        Returns:
            List[TagCombination]: A list of TagCombination objects containing cluster ID, descriptors,
                                  combination type, tags, and tag attributes.
        """
        tag_combinations = []

        for row in tqdm(self.tag_df.to_dict(orient='records'), total=len(self.tag_df)):
            cluster_id = row['index']
            descriptors = split_descriptors(row['descriptors'])
            comb_type = row['type'].strip()
            tags = split_tags(row['tags'])
            if 'attr' not in comb_type:
                processed_tags = []
                processed_attributes = []
                for tag in tags:
                    for sep in SEPERATORS:
                        if sep in tag:
                            tag_key, tag_value = tag.split(sep)
                            processed_tags.append(Tag(key=tag_key, operator=sep, value=tag_value))

                            tag_attributes = self.request_related_tag_attributes(tag_key=tag_key,
                                                                                 tag_value=tag_value,
                                                                                 limit=self.att_limit)
                            processed_attributes.extend(tag_attributes)

            processed_attributes = remove_duplicate_tag_attributes(processed_attributes)
            tag_combinations.append(
                TagCombination(cluster_id=cluster_id, descriptors=descriptors, comb_type=comb_type, tags=processed_tags,
                               tag_attributes=processed_attributes))
        return tag_combinations


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''

    parser = ArgumentParser()
    parser.add_argument('--source', help='domain-specific primary keys', required=True)
    parser.add_argument('--output_file', help='Path to save the tag list', required=True)
    parser.add_argument('--att_limit', help='Enter the number of examples to be fetched by taginfo', default=100)
    parser.add_argument('--att_example_limit', help='Enter the number of examples of the attributes', default=100)
    parser.add_argument('--generate_tag_list_with_attributes', help='Generate tag list with attributes',
                        action='store_true')
    parser.add_argument('--generate_attribute_examples', help='Generate attribute examples',
                        action='store_true')

    args = parser.parse_args()

    source = args.source
    att_limit = args.att_limit
    att_example_limit = args.att_example_limit
    output_file = args.output_file
    generate_tag_list_with_attributes = args.generate_tag_list_with_attributes
    generate_attribute_examples = args.generate_attribute_examples

    comb_retriever = CombinationRetriever(source=source, att_limit=att_limit)

    if generate_tag_list_with_attributes:
        tag_combinations = comb_retriever.generate_tag_list_with_attributes()
        write_output(generated_combs=tag_combinations, output_file=output_file)

    if generate_attribute_examples:
        att_examples = comb_retriever.generate_attribute_examples(num_examples=att_example_limit)
        write_output(generated_combs=att_examples, output_file=output_file)
