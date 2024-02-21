import os
import pandas as pd
import taginfo.query as ti

import urllib
import urllib.request
import urllib.parse
import json
import unicodedata

def json_response_from_url(url):
    '''
    Takes URL and sends request to website, returns the JSON response.

    :param str url: The URL to get a JSON response from
    '''
    url = url.replace(" ", "%20")
    try:
        data = urllib.request.urlopen(url).read()
        return json.loads(data)
    except UnicodeEncodeError:
        print("failed to process", url)
        raise
    except urllib.error.URLError as e:
        print(url)
        if "connection timed out"  in str(e).lower():
            return json_response_from_url(url)
        else:
            raise

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

def retrieve_combinations(source, output_path):
    '''
    Check if a given string only contains english characters. This is meant to filter out values of the name tag
    in different languages. For tags that can have any arbitrary value (e.g. "name") or attribute tags with many
    categorical values (e.g. building:material), a list of possible (example) values is saved as a separate file.

    IMPORTANT: The language limitation should be reworked and adapted in the final database, as more characters
    should be searchable. It should still be discussed what to do with entirely different alphabets.

    :param str source: The path to the source CSV file containing the tag information
    :param str output_path: The path where the output file should be saved
    '''
    tag_df = pd.read_csv(source, header=0, index_col=0, names=['index', 'descriptor', 'type', 'tags', 'combinations'])
    arbitrary_value_df = pd.DataFrame(columns=['key', 'value_list'])

    arbitrary_value_counter = 0 # Keeps track of how many rows were already added to the arbitrary_value_df

    bundle_list = tag_df.loc[(tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'tags'].tolist()
    tag_list = [tag.strip() for candidate in bundle_list for tag in candidate.split(",") if not any(t in tag for t in ["*", "[", " AND "]) or any(t in tag for t in ["***any***", "***numeric***"])]
    # key_list = tag_df.loc[(tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'key'].tolist()
    # value_list = tag_df.loc[(tag_df['type'] == 'core') | (tag_df['type'] == 'core/attr'), 'value'].tolist()
    # tag_list = ["{}={}".format(a, b) for a, b in zip(key_list, value_list)]
    bundle_list_arb = tag_df.loc[(tag_df['type'] == 'attr') | (tag_df['type'] == 'core/attr'), 'tags'].tolist()
    arbitrary_tag_list = [tag.strip() for candidate in bundle_list_arb for tag in candidate.split(",") if not any(t in tag for t in ["*", "[", " AND "]) or any(t in tag for t in ["***any***", "***numeric***"]) ]
    arbitrary_tag_list = [tag.split("=")[0] + "=" for tag in arbitrary_tag_list]
    # key_list_arb = tag_df.loc[(tag_df['type'] == 'attr') | (tag_df['type'] == 'core/attr'), 'key'].tolist()
    # value_list_arb = tag_df.loc[(tag_df['type'] == 'attr') | (tag_df['type'] == 'core/attr'), 'value'].tolist()
    # arbitrary_tag_list = ["{}=".format(a) for a, b in zip(key_list_arb, value_list_arb)]

    for index, row in tag_df.iterrows():
        # key = row['key']
        # value = row['value']
        curr_bundle_list = row['tags'].split(",")
        curr_tag_list = [tag.strip() for tag in curr_bundle_list] # if not any(t in tag for t in ["*", "[", " AND "]) or any(t in tag for t in ["***any***", "***numeric***"])]
        if len(curr_tag_list) == 0:
            continue

        curr_keys = [key.split("=")[0] for key in curr_tag_list]
        curr_vals = [val.split("=")[1] for val in curr_tag_list]
        for key, value in zip(curr_keys, curr_vals):

            if row['type'] != 'core':
                if value == "***any***":
                    arbitrary_value_list = []
                    for i in range (1, 50): # Currently limit the collection of values for e.g. "name" to 6 pages, as retrieving all might take forever
                        value_list = ti.get_page_of_key_values(key, i)
                        for value in value_list:
                            if isRoman(value['value']):
                                arbitrary_value_list.append(value['value'])

                    arbitrary_value_df.loc[arbitrary_value_counter] = [key, '|'.join(arbitrary_value_list)]
                    arbitrary_value_counter += 1

                elif "|" in value:
                    arbitrary_value_df.loc[arbitrary_value_counter] = [key, value]
                    arbitrary_value_counter += 1

            else:
                url = "https://taginfo.openstreetmap.org/api/4/tag/combinations?key=" + urllib.parse.quote(
                    key) + "&value=" + urllib.parse.quote(value)
                data = list(reversed(json_response_from_url(url)['data'])) # Send request to openstreetmap taginfo API to get all possible combinations of a given tag
                combination_list = []
                for item in data: # Search the possible combinations, if they are also part of the local tag database, append them as possible combinations
                    other_tag = item['other_key'] + '=' + item['other_value']
                    other_tag_arbitrary = item['other_key'] + '='

                    if other_tag in tag_list:
                        # matching_row = tag_df[
                        #     (tag_df['key'] == item['other_key']) & (tag_df['value'] == item['other_value'])]
                        matching_row = tag_df[tag_df['tags'].str.contains(other_tag)]
                        corresponding_index = matching_row.index[0]
                        combination_list.append(corresponding_index)

                        # print(key, " - ", value, " - ", matching_row["tags"].values[0])
                    elif other_tag_arbitrary in arbitrary_tag_list and isRoman(item['other_value']):
                        # matching_row = tag_df[(tag_df['key'] == item['other_key'])]
                        matching_row = tag_df[tag_df['tags'].str.contains(other_tag_arbitrary)]
                        corresponding_index = matching_row.index[0]
                        combination_list.append(corresponding_index)

                        # print(key, " - ", value, " - ", matching_row["tags"].values[0])

                if len(combination_list) > 0:
                    combination_list = list(set(combination_list))
                    string_combination = '|'.join(str(number) for number in combination_list)

                    tag_df.at[index, 'combinations'] = string_combination

    tag_df.to_csv(output_path + "Tag_List_v9.csv")
    arbitrary_value_df.to_csv(output_path + "Arbitrary_Value_List_v9.csv")
    print("Saved all files!")

if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    os.makedirs('data', exist_ok=True)

    source = 'data/Primary_Keys_filtered5.csv'
    output_path = 'data/'

    retrieve_combinations(source, output_path)

