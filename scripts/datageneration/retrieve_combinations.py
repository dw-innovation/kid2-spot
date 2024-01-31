from argparse import ArgumentParser
from tqdm import tqdm
from diskcache import Cache
import pandas as pd
import taginfo.query as ti
import requests
import unicodedata

cache = Cache("tmp")

TAG_INFO_API_ENDPOINT = "https://taginfo.openstreetmap.org/api/4/tag/combinations?key=TAG_KEY&value=TAG_VALUE"


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


class CombinationRetriever(object):
    def __init__(self, source, num_examples):
        tag_df = pd.read_csv(source, index_col=False)
        tag_df.drop_duplicates(subset='descriptors',inplace=True)
        tag_df["index"] = [i for i in range(len(tag_df))]
        self.tag_df = tag_df
        self.num_examples = num_examples

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
        arbitrary_values = []
        bundle_list = self.tag_df.loc[
            (self.tag_df['type'] == 'core') | (self.tag_df['type'] == 'core/attr'), 'tags'].tolist()

        # print(bundle_list)

        tag_list = [tag.strip() for candidate in bundle_list for tag in candidate.split(",") if
                    not any(t in tag for t in ["*", "[", " AND "]) or any(
                        t in tag for t in ["***any***", "***numeric***"])]

        # key_list = tag_df.loc[(tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'key'].tolist()
        # value_list = tag_df.loc[(tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'value'].tolist()
        # tag_list = ["{}={}".format(a, b) for a, b in zip(key_list, value_list)]

        bundle_list_arb = self.tag_df.loc[
            (self.tag_df['type'] == 'attr') | (self.tag_df['type'] == 'core/attr'), 'tags'].tolist()
        arbitrary_tag_list = [tag.strip() for candidate in bundle_list_arb for tag in candidate.split(",") if
                              not any(t in tag for t in ["*", "[", " AND "]) or any(
                                  t in tag for t in ["***any***", "***numeric***"])]
        arbitrary_tag_list = [tag.split("=")[0] + "=" for tag in arbitrary_tag_list]

        # key_list_arb = tag_df.loc[(tag_df['type'] == 'attr') | (tag_df['type'] == 'core/attr'), 'key'].tolist()
        # value_list_arb = tag_df.loc[(tag_df['type'] == 'attr') | (tag_df['type'] == 'core/attr'), 'value'].tolist()
        # arbitrary_tag_list = ["{}=".format(a) for a, b in zip(key_list_arb, value_list_arb)]

        for index, row in tqdm(self.tag_df.iterrows(), total=len(self.tag_df)):
            curr_tag_list = [tag.strip() for tag in row['tags'].split(",") if
                             tag.strip()]  # if not any(t in tag for t in ["*", "[", " AND "]) or any(t in tag for t in ["***any***", "***numeric***"])

            if len(curr_tag_list) == 0:
                continue

            tag_key_value_pairs = list(map(lambda pair: (pair.split('=')[0], pair.split('=')[1]), curr_tag_list))

            if row['type'] == 'core':
                for tag_key, tag_value in tag_key_value_pairs:
                    self.assign_combinations(arbitrary_tag_list, index, tag_key, tag_list, tag_value)

            else:
                for tag_key, tag_value in tag_key_value_pairs:
                    if tag_value == "***any***":
                        arbitrary_value_list = []
                        for i in range(1, self.num_examples):  # Currently limit the collection of values for e.g. "name" to 6 pages, as retrieving all might take forever
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

    def assign_combinations(self, arbitrary_tag_list, index, tag_key, all_tags, tag_value):
        combinations = request_tag_combinations(tag_key=tag_key, tag_value=tag_value)

        if combinations['total'] == 0:
            return None

        combination_list = []

        for combination in combinations[
            'data']:  # Search the possible combinations, if they are also part of the local tag database, append them as possible combinations
            other_tag = combination['other_key'] + '=' + combination['other_value']
            other_tag_arbitrary = combination['other_key'] + '='

            if other_tag in all_tags:
                # matching_row = tag_df[
                #     (tag_df['key'] == item['other_key']) & (tag_df['value'] == item['other_value'])]
                matching_row = self.tag_df[self.tag_df['tags'].str.contains(other_tag)]
                corresponding_index = matching_row.index[0]
                combination_list.append(corresponding_index)

                # print(key, " - ", value, " - ", matching_row["tags"].values[0])
            elif other_tag_arbitrary in arbitrary_tag_list and isRoman(combination['other_value']):
                # matching_row = tag_df[(tag_df['key'] == item['other_key'])]
                matching_row = self.tag_df[self.tag_df['tags'].str.contains(other_tag_arbitrary)]
                corresponding_index = matching_row.index[0]
                combination_list.append(corresponding_index)

                # print(key, " - ", value, " - ", matching_row["tags"].values[0])
        if len(combination_list) > 0:
            combination_list = list(set(combination_list))
            string_combination = '|'.join(str(number) for number in combination_list)
            self.tag_df.at[index, 'combinations'] = string_combination


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''

    parser = ArgumentParser()
    parser.add_argument('--source', help='domain-specific primary keys', required=True)
    parser.add_argument('--tag_list', help='Path to save the tag list', required=True)
    parser.add_argument('--num_examples', help='Enter the number of examples to be fetched by taginfo', default=50)
    parser.add_argument('--arbitrary_value_list', help='Path to save the tag list', required=True)

    args = parser.parse_args()

    source = args.source
    tag_list = args.tag_list
    arbitrary_value_list = args.arbitrary_value_list

    CombinationRetriever(source=source, num_examples=args.num_examples).run(tag_list_file=tag_list, arbitrary_value_list_file=arbitrary_value_list)
