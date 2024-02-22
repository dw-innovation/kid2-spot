from argparse import ArgumentParser
from itertools import product

import pandas as pd
from datageneration.utils import write_output
from tqdm import tqdm


def generate_condition(key, operator, value):
    return {
        "key": key.strip(" \"[]"),
        "operator": operator,
        "value": value.strip(" \"[]")
    }


def generate_and_condition(*conditions):
    # cdt_list = []
    # for cdt in conditions:
    #     cdt_list.append([list(c_) for c_ in cdt])
    # return {"and": cdt_list}
    if len(conditions) > 1:
        print("Error: length of contitions > 0, please check!")
    and_list = [list(c_) for c_ in conditions[0]]
    return {"and": [l_[0] for l_ in and_list]}


def generate_or_condition(*conditions):
    if len(conditions[0]) > 1:
        d = {"or": conditions[0]}
        return d
    else:
        return conditions[0]


def yield_tag_flts(tags, yield_final=True):
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        if len(tag) == 0:
            continue

        if "AND" in tag:
            and_list = [t_.strip() for t_ in tag.split('AND')]
            flt_list = [yield_tag_flts(al) for al in and_list]
            yield generate_and_condition(flt_list)
        else:
            op = "="
            if "~" in tag:
                op = "~"
            elif ">" in tag:
                op = ">"
            elif "<" in tag:
                op = "<"
            elif "!=" in tag:
                op = "!="

            key = tag.split(op)[0]
            val = tag.split(op)[1]

            def split_and_list(input):
                if "|" in input:
                    item = input.strip(" \"[]")
                    item_list = item.split("|")
                    item = []
                    for i_ in item_list:
                        item.append(i_)
                    return item
                else:
                    return [input]

            key = split_and_list(key)
            val = split_and_list(val)

            for comb in list(product(key, val)):
                if yield_final:
                    yield generate_condition(comb[0], op, comb[1])


if __name__ == '__main__':
    '''
    Load the current tag bundle list, and transform it to a version in which all tag bundles are represented in the
    graph database format the model translates natural sentences into. Save the result as JSON.
    '''

    parser = ArgumentParser()
    parser.add_argument('--tag_list_path', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    output_file = args.output_file
    tag_list_path = args.tag_list_path

    tag_df = pd.read_csv(tag_list_path)
    tag_df = tag_df[tag_df.select_dtypes(float).notna().any(axis=1)]

    result_dict = {}
    for row in tqdm(tag_df.to_dict(orient='records'), total=len(tag_df)):

        descriptor = row['descriptors']
        tags = row['tags']

        if "," in tags:
            tags = [t_.strip() for t_ in tags.split(',')]
        else:
            tags = [tags]

        result = []
        if len(tags) > 0:
            result += [generate_or_condition(list(yield_tag_flts(tags)))]
        else:
            # result = [og_tag]
            print("ERROR! No tag found!")
        # data_dict[descriptor] = result

        for d_ in descriptor.split("|"):
            if d_ in result_dict:
                print("Error! Duplicate descriptor: ", d_)
            else:
                result_dict[d_] = result
                # result_list.append({"imr": result, "applies_to": descriptor})

    write_output([{"keyword": keyword, "imr": imr} for keyword, imr in result_dict.items()], output_file)
