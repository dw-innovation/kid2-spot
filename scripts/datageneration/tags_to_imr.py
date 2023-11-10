import csv
import json
from itertools import product

from generate_combination_table import NpEncoder

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
    return {"and": [list(c_) for c_ in conditions[0]]}

def generate_or_condition(*conditions):
    if len(conditions[0]) > 1:
        d = {"or": conditions[0]}
        return d
    else:
        return conditions[0]

def main():
    '''
    Load the current tag bundle list, and transform it to a version in which all tag bundles are represented in the
    graph database format the model translates natural sentences into. Save the result as JSON.
    '''
    result_dict = {}
    # result_list = []

    csv_file_path = 'data/Tag_List_v9.csv'
    with (open(csv_file_path, mode='r', newline='') as file):
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # og_tag = generate_condition(row["key"], "=", row["value"])
            descriptor = row['descriptor']
            tags = row['tags']

            def yield_tag_flts(tags, yield_final=True):
                if "," in tags:
                    tag_list = [t_.strip() for t_ in tags.split(',')]
                else:
                    tag_list = [tags]

                for tag in tag_list:
                    if "AND" in tag:
                        and_list = [t_.strip() for t_ in tag.split('AND')]

                        flt_list = [yield_tag_flts(al) for al in and_list]
                        yield generate_and_condition(flt_list)
                    else:
                        op = "="
                        if ">" in tag:
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

    with open("data/imr-tag-db_v1.json", "w") as jsonfile:
        # json.dump(data_dict, jsonfile, cls=NpEncoder)
        json.dump(result_dict, jsonfile, cls=NpEncoder)
        print("Saved files to output path!")

main()