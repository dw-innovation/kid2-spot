import pandas as pd
import json
from argparse import ArgumentParser
from itertools import chain, product
from tqdm import tqdm
from typing import List, Dict, Union

from datageneration.data_model import Tag
from datageneration.utils import write_output, SEPERATORS, split_descriptors, write_dict_output


def generate_and_condition(conditions: List) -> Dict[str, List[Tag]]:
    """
    Generate an 'AND' condition from a list of conditions.

    This function takes a list of conditions, where each condition is a list of Tag objects,
    and combines them into a single 'AND' condition.

    :param conditions: A list of conditions, where each condition is represented as a list of Tag objects.
    :return: A dictionary containing the 'AND' condition, with the key 'and' mapped to a list of Tag objects.
    """
    res = {"and": list(chain.from_iterable(conditions))}
    return res


def generate_or_condition(conditions: List) -> Union[List[Tag], Dict[str, List[Tag]]]:
    """
    Generate an 'OR' condition from a list of conditions.

    This function takes a list of conditions, where each condition is a list of Tag objects,
    and combines them into a single 'OR' condition.

    :param conditions: A list of conditions, where each condition is represented as a list of Tag objects.
    :return: A dictionary containing the 'AND' condition, with the key 'and' mapped to a list of Tag objects.
    """
    first_condition = conditions[0]

    if isinstance(first_condition, Tag) or len(conditions) > 1:
        return {"or": conditions}

    if isinstance(first_condition, dict):
        return first_condition

    return conditions[0]


def transform_tags_to_imr(tags_str: str) -> List[Dict[str, List[Tag]]]:
    '''
    Transform tag lists in a string format into IMR which contains tag filters
    :param tags_str:
    :return: list of dictionary
    '''
    if "," in tags_str:
        tags = [t_.strip() for t_ in tags_str.split(',')]
    else:
        tags = [tags_str]

    result = []
    if tags:
        result.append(generate_or_condition(list(yield_tag_filters_for_imr(tags))))
    return result if isinstance(result[0], list) else result


def yield_tag_filters_for_imr(tags: Union[str, List[str]]) -> List[Tag]:
    """
    Yield tag filters for constructing IMR. Filters are connected each other AND or OR operators

    :param tags (str or list of str): The tag string or list of tag strings to be processed.

    :return: list of tags: a list of tags
    """
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        if not tag:
            continue
        if "AND" in tag:
            and_list = [t_.strip() for t_ in tag.split('AND')]
            flt_list = [yield_tag_filters_for_imr(al) for al in and_list]
            yield generate_and_condition(flt_list)
        else:
            op = next((o for o in SEPERATORS if o in tag), "=")
            tag_key, tag_value = tag.split(op)
            tag_key = [k.strip(" \"[]") for k in tag_key.split("|")]
            tag_value = [v.strip(" \"[]") for v in tag_value.split("|")]

            for comb in product(tag_key, tag_value):
                yield Tag(key=comb[0], operator=op, value=comb[1])

def tag_serializer(tag):
    return tag.to_dict()

if __name__ == '__main__':
    '''
    Load the current tag bundle list, and transform it to a version in which all tag bundles are represented in the
    graph database format the model translates natural sentences into. Save the result as JSON.
    '''

    parser = ArgumentParser()
    parser.add_argument('--primary_key_table', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    output_file = args.output_file
    tag_list_path = args.primary_key_table

    primary_key_table = pd.read_excel(args.primary_key_table, engine='openpyxl')

    results = []
    for row in tqdm(primary_key_table.to_dict(orient='records'), total=len(primary_key_table)):
        descriptors_str = row['descriptors']
        tags_str = row['tags']

        desriptors = split_descriptors(descriptors_str)
        tags = json.loads(json.dumps(transform_tags_to_imr(tags_str), default=tag_serializer))

        for descriptor in desriptors:
            results.append(dict(key=descriptor, imr=tags))
    write_dict_output(results, output_file, bool_add_yaml=False)
