import unicodedata
from argparse import ArgumentParser
from typing import List

import pandas as pd
import requests
import taginfo.query as ti
from datageneration.data_model import TagAttribute
from datageneration.utils import CompoundTagAttributeProcessor, SEPERATORS
from diskcache import Cache
from tqdm import tqdm

cache = Cache("tmp")

TAG_INFO_API_ENDPOINT = "https://taginfo.openstreetmap.org/api/4/tag/combinations?key=TAG_KEY&value=TAG_VALUE&filter=nodes&sortname=together_count&sortorder=desc"


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


# def isEnglish(s):
#     '''
#     Check if a given string only contains english characters. This is meant to filter out values of the name tag
#     in different languages.
#     IMPORTANT: This should be reworked and adapted in the final database, as more characters should be searchable.
#
#     :param str s: The string to be checked
#     '''
#     try:
#         s.encode(encoding='utf-8').decode('ascii')
#     except UnicodeDecodeError:
#         return False
#     else:
#         return True


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


class CombinationRetriever(object):
    def __init__(self, source, num_examples):
        if source.endswith('xlsx'):
            tag_df = pd.read_excel(source, engine='openpyxl')
        else:
            tag_df = pd.read_csv(source, index_col=False)

        tag_df.drop_duplicates(subset='descriptors', inplace=True)
        tag_df["index"] = [i for i in range(len(tag_df))]

        all_tags = {}
        for tag in tag_df.to_dict(orient='records'):
            all_tags[int(tag['index'])] = tag

        all_osm_tags_and_attributes = self.process_tag_attributes(tag_df)
        self.all_osm_tags_and_attributes = all_osm_tags_and_attributes
        self.all_tags = all_tags
        self.tag_df = tag_df
        self.num_examples = num_examples

    def process_tag_attributes(self, tag_df):
        # all tags and attributes
        all_osm_tags_and_attributes = {}
        for tags in tag_df.to_dict(orient='records'):
            tag_type = tags['type']
            if isinstance(tag_type, float):
                print(f'{tags} has no type, might be an invalid')
                continue
            tag_type = tag_type.strip()
            for tag in tags['tags'].split(','):
                tag = tag.lstrip().strip()
                if 'AND' in tag:
                    _tags = tag.split('AND')
                    for _tag in _tags:
                        _tag = _tag.lstrip().strip().replace(' ', '').lower()
                        if _tag not in all_osm_tags_and_attributes.keys():
                            all_osm_tags_and_attributes[_tag] = {'tag': _tag, 'type': tag_type}
                else:
                    tag = tag.replace(' ', '').lower()
                    if len(tag) == 0:
                        continue

                    if '[' in tag:
                        compound_tag = comp_att_processor.run(tag)
                        for alt_tag in compound_tag:
                            if alt_tag not in all_osm_tags_and_attributes.keys():
                                all_osm_tags_and_attributes[alt_tag] = {'tag': alt_tag, 'type': tag_type}
                    else:
                        if tag not in all_osm_tags_and_attributes.keys():
                            all_osm_tags_and_attributes[tag] = {'tag': tag, 'type': tag_type}
        return all_osm_tags_and_attributes

    def run(self, tag_list_file, arbitrary_value_list_file):
        '''
        Check if a given string only contains english characters. This is meant to filter out values of the name tag
        in different languages. For tags that can have any arbitrary value (e.g. "name") or attribute tags with many
        categorical values (e.g. building:material), a list of possible (example) values is saved as a separate file.

        IMPORTANT: The language limitation should be reworked and adapted in the final database, as more characters
        should be searchable. It should still be discussed what to do with entirely different alphabets.

        :param str source: The path to the source CSV file containing the tag information
        :param str tag_list_file: The path where the taglist file is saved
        :param list arbitrary_value_list_file: The path where the arbitrary value list is saved
        '''
        # print('tag list file=====')
        # print(tag_list_file)
        arbitrary_values = []
        bundle_list = self.tag_df.loc[
            (self.tag_df['type'] == 'core') | (self.tag_df['type'] == 'core/attr'), 'tags'].tolist()

        # print(bundle_list)

        tag_list = [tag.strip() for candidate in bundle_list for tag in candidate.split(",") if
                    not any(t in tag for t in ["*", "[", " AND "]) or any(
                        t in tag for t in ["***any***", "***numeric***"])]
        bundle_list_arb = self.tag_df.loc[
            (self.tag_df['type'] == 'attr') | (self.tag_df['type'] == 'core/attr'), 'tags'].tolist()

        # what is difference between tag list and arbitrary tag list
        arbitrary_tag_list = [tag.strip() for candidate in bundle_list_arb for tag in candidate.split(",") if
                              not any(t in tag for t in ["*", "[", " AND "]) or any(
                                  t in tag for t in ["***any***", "***numeric***"])]

        arbitrary_tag_list = [tag.split("=")[0] + "=" for tag in arbitrary_tag_list]
        for idx, row in tqdm(self.all_tags.items(), total=len(self.all_tags)):
            associated_tags = [tag.strip() for tag in row['tags'].split(",") if
                               tag.strip()]  # if not any(t in tag for t in ["*", "[", " AND "]) or any(t in tag for t in ["***any***", "***numeric***"])

            if len(associated_tags) == 0:
                print(f"No assocated tags found for {row['descriptors']}")
                continue

            tag_key_value_pairs = []
            for pair in associated_tags:
                if '=' in pair:
                    tag_key_value_pairs.append((pair.split('=')[0], pair.split('=')[1]))
                elif '~' in pair:
                    tag_key_value_pairs.append((pair.split('~')[0], pair.split('~')[1]))

            if row['type'] == 'core':
                for tag_key, tag_value in tag_key_value_pairs:
                    combinations = self.assign_combinations(arbitrary_tag_list, tag_key, tag_list, tag_value)

                    if combinations:
                        self.tag_df.at[row['index'], 'combinations'] = combinations
            else:
                for tag_key, tag_value in tag_key_value_pairs:
                    if tag_value == "***any***":
                        arbitrary_value_list = []
                        for i in range(1,
                                       self.num_examples):  # Currently limit the collection of values for e.g. "name" to 6 pages, as retrieving all might take forever
                            value_list = ti.get_page_of_key_values(tag_key, i)
                            for tag_value in value_list:
                                if isRoman(tag_value['value']):
                                    arbitrary_value_list.append(tag_value['value'])

                        arbitrary_values.append(
                            {"key": tag_key, "value_list": [tag_key, '|'.join(arbitrary_value_list)]})

                    elif "|" in tag_value:
                        arbitrary_values.append(
                            {"key": tag_key, "value_list": tag_value})

        self.tag_df.to_csv(tag_list_file, index=False)
        pd.DataFrame(arbitrary_values).to_csv(arbitrary_value_list_file, index=False)
        print("Saved all files!")

    def index_to_descriptors(self, index):
        return self.all_tags[index]['descriptors']

    def request_related_tag_attributes(self, tag_key: str, tag_value: str, limit: str = 100) -> List[TagAttribute]:
        combinations = request_tag_combinations(tag_key=tag_key, tag_value=tag_value)['data']
        selected_attributes = set()
        all_tags_attributes_ids = self.all_osm_tags_and_attributes.keys()
        for combination in combinations:
            if len(selected_attributes) == limit:
                print(f"Number of selected attributes {len(selected_attributes)}")
                return list(selected_attributes)
            for seperator in SEPERATORS:
                other_tag = combination['other_key'] + seperator + combination['other_value']
                print(f"Searching {other_tag}")
                if other_tag in all_tags_attributes_ids:
                    print(f"The tag {other_tag} in the list")
                    other_tag_type = self.all_osm_tags_and_attributes[other_tag]['type']
                    if other_tag_type == 'core':
                        print(f"{other_tag} is filtered out, since its type is {other_tag_type}.")
                    else:
                        selected_attributes.add(self.all_osm_tags_and_attributes[other_tag]['tag'])
                        print(f"{other_tag} is in our list and its type is {other_tag_type}.")
                        continue
                else:
                    print(f"The tag {other_tag} is not in our list but maybe rephrased version might be in our list")
                    results = list(filter(lambda x: x.startswith(f"{other_tag}"), all_tags_attributes_ids))

                    print("Number of results is ", len(results))
                    if len(results) == 0:
                        rewritten_tag = other_tag.split(seperator)[0]
                        results = list(
                            filter(lambda x: x.startswith(f"{rewritten_tag}{seperator}"), all_tags_attributes_ids))

                        if len(results) == 0:
                            print(f"No result found for {other_tag} even though rewritten tag is {rewritten_tag}")
                            continue

                        for result in results:
                            print(f"Found result for {other_tag} which was rewritten as {rewritten_tag}")
                            attribute_value = result
                            other_tag_type = self.all_osm_tags_and_attributes[attribute_value]['type']
                            if other_tag_type == 'core':
                                print(f"{attribute_value} is a core, so we exclude.")
                            else:
                                selected_attributes.add(attribute_value)
                                print(f"{attribute_value} is type of {other_tag_type}, we include it.")
                        continue

                    else:
                        for result in results:
                            print(f"Found result for {other_tag}")
                            attribute_value = result
                            other_tag_type = self.all_osm_tags_and_attributes[attribute_value]['type']
                            if other_tag_type == 'core':
                                print(f"{attribute_value} is a core, so we exclude.")
                            else:
                                selected_attributes.add(attribute_value)
                                print(f"{attribute_value} is type of {other_tag_type}, we include it.")

        print(f"Number of selected attributes {len(selected_attributes)}")
        print(selected_attributes)
        return list(selected_attributes)

    def assign_combinations(self, arbitrary_tag_list, tag_key, all_tags, tag_value):
        print(tag_key)
        print(tag_value)
        combinations = request_tag_combinations(tag_key=tag_key, tag_value=tag_value)

        if combinations['total'] == 0:
            return None

        filtered_combinations = []
        for combination in combinations[
            'data']:  # Search the possible combinations, if they are also part of the local tag database, append them as possible combinations

            other_tag = combination['other_key'] + '=' + combination['other_value']
            other_tag_arbitrary = combination['other_key'] + '='

            if other_tag in all_tags:
                try:
                    matching_row = self.tag_df[self.tag_df['tags'].str.contains(other_tag)]
                except ValueError as e:
                    print(f"combination: {combination}, other tag {other_tag} got an exception: {e}")
                    continue

                for matched_index in matching_row['index'].tolist():
                    filtered_combinations.append(matched_index)

            elif other_tag_arbitrary in arbitrary_tag_list and isRoman(combination['other_value']):
                try:
                    matching_row = self.tag_df[self.tag_df['tags'].str.contains(other_tag_arbitrary)]
                except ValueError as e:
                    print(f"combination: {combination}, other tag {other_tag_arbitrary} got an exception: {e}")
                    continue

                # ipek changes:
                for matched_index in matching_row['index'].tolist():
                    filtered_combinations.append(matched_index)

                # print(key, " - ", value, " - ", matching_row["tags"].values[0])
        if len(filtered_combinations) > 0:
            filtered_combinations = list(set(filtered_combinations))
            string_combination = '|'.join(str(number) for number in filtered_combinations)
            return string_combination

        return None


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''

    parser = ArgumentParser()
    parser.add_argument('--source', help='domain-specific primary keys', required=True)
    parser.add_argument('--tag_list', help='Path to save the tag list', required=True)
    parser.add_argument('--num_examples', help='Enter the number of examples to be fetched by taginfo', default=50)
    parser.add_argument('--arbitrary_value_list', help='Path to save the tag list', required=True)
    parser.add_argument('--generate_tag_list_with_attributes', help='Generate tag list with attributes',
                        action='store_true')

    args = parser.parse_args()

    source = args.source
    tag_list = args.tag_list
    arbitrary_value_list = args.arbitrary_value_list
    generate_tag_list_with_attributes = args.generate_tag_list_with_attributes

    comb_retriever = CombinationRetriever(source=source, num_examples=args.num_examples)

    if generate_tag_list_with_attributes:
        comb_retriever.generate_tag_list_with_attributes()
